"""One GPU session: preflight, provision, sync, run scenarios, collect, teardown, verify.

Teardown runs in a `finally`, so a failed setup, a failed scenario, Ctrl-C or SIGTERM still
terminates the instance and deletes the security group and key pair. Afterwards the session's
resources are looked up twice, through the Resource Groups Tagging API (`session=<id>`) and by
the ids the session recorded, and any that still exist fail the session loudly. If this
machine dies instead, the instance shuts itself down after `max_session_minutes` (shutdown
means terminate), and `teardown --session <id>` removes the free leftovers (security group, key
pair) from the tags and the saved state.

The session writes its own results JSON (kind `cloud.gpu_session`) with the instance, AMI,
billed time and cost, the teardown check, and the scenario results it brought back. Each
scenario results file gets its cost (hourly rate x its own wall time) and the session's git SHA
when it was written inside a container without git.
"""

from __future__ import annotations

import json
import math
import shlex
import signal
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vms.bench.results import (
    GitInfo,
    Hardware,
    RunResult,
    git_info,
    package_versions,
    write_result,
)
from vms.bench.results import hardware as local_hardware
from vms.cloud.aws import Cloud, Identity, Image, LaunchDeniedError, LaunchSpec
from vms.cloud.remote import Remote, RemoteError
from vms.cloud.settings import CloudSettings
from vms.common.clock import Clock
from vms.common.logs import get_logger

log = get_logger(__name__)

KIND = "cloud.gpu_session"
REMOTE_SCRIPT = "scripts/gpu_remote.sh"
CLIPS_DIR = "data/clips"
RESULTS_DIR = "bench/results"

# Prints this machine's hardware as JSON on the instance (system python3, no project code yet).
HW_PROBE = """
import json, os, platform, subprocess
cpu = platform.machine()
for line in open("/proc/cpuinfo"):
    if line.startswith("model name"):
        cpu = line.split(":", 1)[1].strip()
        break
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=60).stdout
print(json.dumps({"cpu": cpu, "logical_cpus": os.cpu_count(),
                  "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                  "platform": platform.platform(),
                  "gpus": [g.strip() for g in smi.splitlines() if g.strip()]}))
"""


class PreflightError(Exception):
    """The session cannot start; nothing was created."""


@dataclass(frozen=True)
class Preflight:
    identity: Identity
    image: Image
    instance_type: str
    vcpus: int
    vcpu_quota: float
    hourly_usd: float
    max_session_minutes: int
    existing: list[str]  # project resources from earlier sessions that still exist

    @property
    def max_cost_usd(self) -> float:
        return self.hourly_usd * self.max_session_minutes / 60


@dataclass
class SessionState:
    """What a session created, saved after every step so a crashed run can be torn down."""

    session_id: str
    account: str = ""
    region: str = ""
    key_name: str | None = None
    key_pair_id: str | None = None
    security_group_id: str | None = None
    instance_id: str | None = None
    public_ip: str | None = None
    launched_at: datetime | None = None
    terminated_at: datetime | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), default=str, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SessionState:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("launched_at", "terminated_at"):
            if raw.get(key):
                raw[key] = datetime.fromisoformat(raw[key])
        return cls(**raw)

    def arns(self) -> list[str]:
        prefix = f"arn:aws:ec2:{self.region}:{self.account}"
        pairs = (
            ("instance", self.instance_id),
            ("security-group", self.security_group_id),
            ("key-pair", self.key_pair_id),
        )
        return [f"{prefix}:{kind}/{ident}" for kind, ident in pairs if ident]


@dataclass
class ScenarioRun:
    scenario: str
    ok: bool
    error: str | None = None


@dataclass
class TeardownReport:
    leftovers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return not self.leftovers and not self.errors


def billed_seconds(launched: datetime, terminated: datetime) -> int:
    """EC2 Linux on-demand bills per second with a 60 s minimum."""
    return max(60, math.ceil((terminated - launched).total_seconds()))


def session_tags(settings: CloudSettings, session_id: str) -> dict[str, str]:
    return {**settings.tags, "session": session_id}


class GpuSession:
    def __init__(
        self,
        settings: CloudSettings,
        cloud: Cloud,
        *,
        session_id: str,
        repo: Path,
        state_dir: Path,
        clock: Clock,
        sleep: Callable[[float], None],
        remote_factory: Callable[[str, Path, Path], Remote],
        keygen: Callable[[Path, str], bytes],
        ssh_cidr: Callable[[], str],
        tracked_files: Callable[[Path], list[str]] | None = None,
    ) -> None:
        self.settings = settings
        self.cloud = cloud
        self.session_id = session_id
        self.repo = repo
        self.dir = state_dir / session_id
        self.clock = clock
        self._sleep = sleep
        self._remote_factory = remote_factory
        self._keygen = keygen
        self._ssh_cidr = ssh_cidr
        self._tracked_files = tracked_files or git_tracked_files
        self.state = SessionState(session_id=session_id)
        self.tags = session_tags(settings, session_id)
        self.name = f"vms-gpu-{session_id.lower()}"

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    def _save(self) -> None:
        self.state.save(self.state_path)

    # --- Preflight -----------------------------------------------------------------------

    def _spec(self, image: Image) -> LaunchSpec:
        user_data = (
            "#!/bin/bash\n"
            f'shutdown -h +{self.settings.max_session_minutes} "VMS GPU session time limit"\n'
        )
        return LaunchSpec(
            image=image,
            instance_type=self.settings.instance_type,
            key_name=self.state.key_name or "",
            security_group_id=self.state.security_group_id or "",
            root_volume_gb=self.settings.root_volume_gb,
            user_data=user_data,
            tags={**self.tags, "Name": self.name},
        )

    def preflight(self) -> Preflight:
        """Checks that cost nothing and create nothing: Region, quota, AMI, a dry-run launch."""
        s = self.settings
        identity = self.cloud.identity()
        if s.region and identity.region != s.region:
            raise PreflightError(
                f"AWS Region is {identity.region}, configs/cloud.yaml says {s.region}"
            )
        hourly = s.hourly_usd()
        vcpus = self.cloud.instance_vcpus(s.instance_type)
        quota = self.cloud.vcpu_quota(s.vcpu_quota_code)
        if quota < vcpus:
            raise PreflightError(
                f"{s.instance_type} needs {vcpus} vCPUs but the on-demand G and VT quota "
                f"({s.vcpu_quota_code}) in {identity.region} is {quota:g}: request an increase "
                "in Service Quotas (RUNBOOK, GPU sessions)"
            )
        image = self.cloud.resolve_image(s.ami_parameter)
        try:
            self.cloud.check_launch(self._spec(image))
        except LaunchDeniedError as exc:
            raise PreflightError(f"a dry-run launch was refused: {exc}") from exc
        existing = [arn for arn in self.cloud.tagged_arns(s.tags) if self.cloud.exists(arn)]
        return Preflight(
            identity=identity,
            image=image,
            instance_type=s.instance_type,
            vcpus=vcpus,
            vcpu_quota=quota,
            hourly_usd=hourly,
            max_session_minutes=s.max_session_minutes,
            existing=existing,
        )

    # --- The session ---------------------------------------------------------------------

    def run(
        self,
        scenarios: list[str],
        confirm: Callable[[Preflight], bool],
        results_dir: Path,
        *,
        allow_dirty: bool = False,
    ) -> int:
        """Run a whole session. 0: all good; 1: setup or a scenario failed; 2: leftovers;
        3: aborted before launch (nothing created); 130: interrupted."""
        git = git_info(self.repo)
        if git.sha is None:
            raise PreflightError(f"{self.repo} is not a git checkout")
        if git.dirty and not allow_dirty:
            raise PreflightError(
                "uncommitted changes: commit first so results carry an exact git SHA "
                "(or pass --allow-dirty for a test session)"
            )
        for scenario in scenarios:
            if not (self.repo / scenario).is_file():
                raise PreflightError(f"scenario not found: {scenario}")
        pre = self.preflight()
        if not confirm(pre):
            log.info("session not confirmed; nothing was created")
            return 3

        self.state.account = pre.identity.account
        self.state.region = pre.identity.region
        started = self.clock.now()
        runs: list[ScenarioRun] = []
        collected: list[Path] = []
        remote_hw: Hardware | None = None
        error: str | None = None
        status = 0
        teardown = TeardownReport()
        try:
            remote = self._provision(pre.image)
            remote_hw = self._bootstrap(remote)
            self._sync(remote)
            remote.run(self._remote("setup"), self.settings.timeouts.setup_s)
            for scenario in scenarios:
                runs.append(self._run_scenario(remote, scenario))
            collected = self._collect(remote, results_dir)
            if not all(r.ok for r in runs):
                status = 1
        except KeyboardInterrupt:
            error, status = "interrupted", 130
            log.warning("interrupted: tearing down")
        except Exception as exc:
            error, status = f"{type(exc).__name__}: {exc}", 1
            log.exception("session failed: tearing down")
        finally:
            with shielded():
                teardown = self.teardown()
        cost = self._cost()
        self._annotate(collected, git, pre.hourly_usd)
        record = self._record(
            pre, git, started, scenarios, runs, collected, remote_hw, teardown, error, cost
        )
        path = write_result(results_dir, record)
        log.info("session results written", extra={"path": str(path), "cost_usd": cost})
        if not teardown.verified:
            return 2
        return status

    def _remote(self, *args: str) -> str:
        quoted = " ".join(shlex.quote(a) for a in args)
        return f"cd {shlex.quote(self.settings.remote.dir)} && bash {REMOTE_SCRIPT} {quoted}"

    def _provision(self, image: Image) -> Remote:
        t = self.settings.timeouts
        key_path = self.dir / "id_ed25519"
        public_key = self._keygen(key_path, self.name)
        self.state.key_name = self.name
        self.state.key_pair_id = self.cloud.import_key_pair(self.name, public_key, self.tags)
        self._save()
        self.state.security_group_id = self.cloud.create_security_group(
            self.name, self._ssh_cidr(), self.tags
        )
        self._save()
        self.state.instance_id = self.cloud.run_instance(self._spec(image))
        self._save()
        log.info("instance launched", extra={"instance_id": self.state.instance_id})
        instance = self._wait_instance(
            lambda i: i is not None and i.state == "running", t.running_s
        )
        assert instance is not None
        if not instance.public_ip:
            raise RuntimeError("the instance has no public IP (default subnet without one?)")
        self.state.public_ip = instance.public_ip
        self.state.launched_at = instance.launch_time or self.clock.now()
        self._save()
        remote = self._remote_factory(instance.public_ip, key_path, self.dir / "known_hosts")
        remote.wait_ready(t.ssh_ready_s)
        return remote

    def _wait_instance(self, done: Callable[[Any], bool], timeout_s: float) -> Any:
        assert self.state.instance_id is not None
        deadline = self.clock.monotonic() + timeout_s
        while True:
            instance = self.cloud.describe_instance(self.state.instance_id)
            if done(instance):
                return instance
            if self.clock.monotonic() >= deadline:
                state = instance.state if instance else "gone"
                raise TimeoutError(f"instance {self.state.instance_id} still {state}")
            self._sleep(self.settings.timeouts.poll_s)

    def _bootstrap(self, remote: Remote) -> Hardware:
        # The docker group applies to the next SSH login, i.e. every later command.
        remote.run('sudo usermod -aG docker "$USER"', 60, capture=True)
        out = remote.run(f"python3 -c {shlex.quote(HW_PROBE)}", 120, capture=True)
        hw = Hardware.model_validate_json(out.strip().splitlines()[-1])
        if not hw.gpus:
            raise RuntimeError("nvidia-smi found no GPU on the instance")
        log.info("instance hardware", extra={"hardware": hw.model_dump()})
        return hw

    def _sync(self, remote: Remote) -> None:
        paths = [*self._tracked_files(self.repo), ".git"]
        paths += [
            f"{CLIPS_DIR}/{clip}"
            for clip in self.settings.remote.clips
            if (self.repo / CLIPS_DIR / clip).is_dir()
        ]
        remote.push(self.repo, paths, self.settings.remote.dir, self.settings.timeouts.sync_s)

    def _run_scenario(self, remote: Remote, scenario: str) -> ScenarioRun:
        log.info("scenario starting", extra={"scenario": scenario})
        try:
            remote.run(self._remote("run", scenario), self.settings.timeouts.scenario_s)
        except RemoteError as exc:
            log.error("scenario failed", extra={"scenario": scenario, "error": str(exc)})
            return ScenarioRun(scenario, ok=False, error=str(exc))
        return ScenarioRun(scenario, ok=True)

    def _collect(self, remote: Remote, results_dir: Path) -> list[Path]:
        """Copy the results files the session produced (not those synced from the repo)."""
        staging = self.dir / "results"
        remote.pull(
            f"{self.settings.remote.dir}/{RESULTS_DIR}", staging, self.settings.timeouts.collect_s
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        new: list[Path] = []
        for path in sorted(staging.glob("*.json")):
            target = results_dir / path.name
            if target.exists():
                continue
            target.write_bytes(path.read_bytes())
            new.append(target)
        return new

    # --- Teardown ------------------------------------------------------------------------

    def adopt_tagged(self) -> None:
        """Fill in resource ids from the session's tags (for `teardown` of a crashed run)."""
        for arn in self.cloud.tagged_arns(self.tags):
            parts = arn.split(":")
            self.state.region = self.state.region or parts[3]
            self.state.account = self.state.account or parts[4]
            kind, _, ident = parts[5].partition("/")
            if kind == "instance" and not self.state.instance_id:
                self.state.instance_id = ident
            elif kind == "security-group" and not self.state.security_group_id:
                self.state.security_group_id = ident
            elif kind == "key-pair" and not self.state.key_pair_id:
                self.state.key_pair_id = ident

    def teardown(self) -> TeardownReport:
        """Delete everything the session created, then check nothing tagged with it remains.
        Each step runs even if an earlier one failed."""
        t = self.settings.timeouts
        report = TeardownReport()
        if self.state.instance_id:
            try:
                self.cloud.terminate_instance(self.state.instance_id)
                self._wait_instance(lambda i: i is None or i.state == "terminated", t.terminate_s)
                self.state.terminated_at = self.clock.now()
            except Exception as exc:
                report.errors.append(f"terminate {self.state.instance_id}: {exc}")
            self._save()
        if self.state.security_group_id:
            try:
                self._delete_security_group(self.state.security_group_id)
            except Exception as exc:
                report.errors.append(f"delete {self.state.security_group_id}: {exc}")
        if self.state.key_pair_id:
            try:
                self.cloud.delete_key_pair(self.state.key_pair_id)
            except Exception as exc:
                report.errors.append(f"delete {self.state.key_pair_id}: {exc}")
        self._save()
        try:
            report.leftovers = self.leftovers()
        except Exception as exc:
            report.errors.append(f"leftover check: {exc}")
        if report.verified:
            log.info("teardown verified: nothing tagged with this session remains")
        else:
            log.error(
                "TEARDOWN NOT VERIFIED: resources may still be billing",
                extra={"leftovers": report.leftovers, "errors": report.errors},
            )
        return report

    def _delete_security_group(self, group_id: str) -> None:
        # The instance's network interface can outlive "terminated" by a little while.
        deadline = self.clock.monotonic() + self.settings.timeouts.delete_retry_s
        while not self.cloud.delete_security_group(group_id):
            if self.clock.monotonic() >= deadline:
                raise TimeoutError(f"security group {group_id} still in use")
            self._sleep(self.settings.timeouts.poll_s)

    def leftovers(self) -> list[str]:
        """Resources tagged with this session, or recorded by it, that still exist."""
        candidates = set(self.cloud.tagged_arns(self.tags)) | set(self.state.arns())
        return sorted(arn for arn in candidates if self.cloud.exists(arn))

    # --- Results -------------------------------------------------------------------------

    def _billed(self) -> int:
        if self.state.launched_at is None:
            return 0
        end = self.state.terminated_at or self.clock.now()  # not confirmed: bill until now
        return billed_seconds(self.state.launched_at, end)

    def _cost(self) -> float:
        return round(self.settings.hourly_usd() * self._billed() / 3600, 4)

    def _annotate(self, files: list[Path], git: GitInfo, hourly: float) -> None:
        for path in files:
            try:
                result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue  # not a results file
            wall = (result.ended_at - result.started_at).total_seconds()
            note = f"GPU session {self.session_id} ({self.settings.instance_type})"
            if result.git.sha is None:
                result.git = git  # written in a container without git; the session's checkout
                note += "; git from the session"
            result.cost_usd = round(hourly * wall / 3600, 4)
            result.notes = f"{result.notes}; {note}" if result.notes else note
            path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def _record(
        self,
        pre: Preflight,
        git: GitInfo,
        started: datetime,
        scenarios: list[str],
        runs: list[ScenarioRun],
        collected: list[Path],
        remote_hw: Hardware | None,
        teardown: TeardownReport,
        error: str | None,
        cost: float,
    ) -> RunResult:
        s = self.state
        return RunResult(
            run_id=self.session_id,
            name="gpu-session",
            kind=KIND,
            started_at=started,
            ended_at=self.clock.now(),
            seed=self.settings.seed,
            git=git,
            hardware=remote_hw or local_hardware(),
            packages=package_versions(("boto3",)),
            config=self.settings.model_dump(mode="json"),
            scenario={"scenarios": scenarios},
            metrics={
                "region": s.region,
                "instance_type": self.settings.instance_type,
                "instance_id": s.instance_id,
                "image": {"id": pre.image.id, "name": pre.image.name},
                "launched_at": s.launched_at.isoformat() if s.launched_at else None,
                "terminated_at": s.terminated_at.isoformat() if s.terminated_at else None,
                "billed_s": self._billed(),
                "hourly_usd": round(pre.hourly_usd, 4),
                "scenarios": [asdict(r) for r in runs],
                "results_files": [p.name for p in collected],
                "error": error,
                "teardown_verified": teardown.verified,
                "leftovers": teardown.leftovers,
                "teardown_errors": teardown.errors,
                "hardware_is": "instance" if remote_hw else "operator machine (no instance)",
            },
            cost_usd=cost,
            notes="cost estimated from configs/cloud.yaml prices; billing data lags a day",
        )


@contextmanager
def shielded() -> Iterator[None]:
    """Ignore Ctrl-C and SIGTERM while tearing down: a second Ctrl-C must not leave the
    instance running. Every step of the teardown has its own timeout."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def git_tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=True, timeout=30
    ).stdout.decode()
    return [p for p in out.split("\0") if p]
