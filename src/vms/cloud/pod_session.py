"""One GPU session on a Runpod pod (D37): preflight, create, sync, run model scenarios, collect,
terminate, verify.

The flow mirrors the EC2 session (`vms.cloud.session`, D28) where the platforms agree and
differs where they do not:

- A pod is a container, not a VM: no Docker or Compose inside it, so only in-process
  scenarios (`kind: model` and `ladder`) run here, in the project's environment with CUDA
  PyTorch. End-to-end scenarios stay on the laptop and the AWS CPU sessions (D35); the GPU
  end-to-end ramp is pending (D37).
- Runpod has no resource tags. The pod's name carries the session id (`<prefix>-gpu-<id>`) and
  the leftover check lists the account's pods by that name, as well as reading the pod id the
  session recorded.
- No key pair or firewall resources: the per-session public key goes in as `PUBLIC_KEY`, which
  the image's start script writes to `authorized_keys`, and sshd is published on `22/tcp`.
- The safety net if this machine dies: the pod's start command arms a timer that deletes the
  pod through the API after `max_session_minutes`, with the pod-scoped `RUNPOD_API_KEY` Runpod
  puts in every pod. The session checks the timer is armed before it runs anything. The prepaid
  balance (auto-pay off) is the last cap.

Teardown runs in a `finally` with signals ignored, as on EC2. The session's results JSON (kind
`cloud.pod_session`) records the pod, GPU, data center, host CUDA version, the catalog price,
the billed time and cost, the teardown check and the scenario results it brought back.
"""

from __future__ import annotations

import json
import math
import shlex
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from vms.bench.results import (
    GitInfo,
    Hardware,
    RunResult,
    git_info,
    package_versions,
    write_result,
)
from vms.bench.results import hardware as local_hardware
from vms.cloud.pod_settings import PodSettings
from vms.cloud.remote import Remote, RemoteError
from vms.cloud.runpod import GpuOffer, Pod, Pods, PodSpec
from vms.cloud.session import (
    HW_PROBE,
    PreflightError,
    ScenarioRun,
    TeardownReport,
    annotate_results,
    collect_new_files,
    git_tracked_files,
    shielded,
)
from vms.common.clock import Clock
from vms.common.logs import get_logger

log = get_logger(__name__)

KIND = "cloud.pod_session"
REMOTE_SCRIPT = "scripts/pod_remote.sh"
CLIPS_DIR = "data/clips"
# A pod has no Docker, so only scenarios that run in one Python process run on it (D37).
POD_SCENARIO_KINDS = frozenset({"model", "ladder"})
GONE = frozenset({"TERMINATED"})
TIMER_LOG = "/var/log/vms-time-limit.log"

# The container's start command: arm the time limit, then hand over to the image's own start
# script (sshd from PUBLIC_KEY). The timer deletes the pod with the pod-scoped key Runpod
# injects, falling back to runpodctl; its first line says whether that key is present.
START_COMMAND = f"""\
(echo "armed pod=${{RUNPOD_POD_ID:-}} key=$([ -n "${{RUNPOD_API_KEY:-}}" ] && echo yes || echo no) \
seconds=$VMS_MAX_SECONDS"
 sleep "$VMS_MAX_SECONDS"
 echo "time limit reached: deleting pod $RUNPOD_POD_ID"
 curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
"https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID"
) >{TIMER_LOG} 2>&1 &
exec /start.sh
"""


@dataclass(frozen=True)
class PodPreflight:
    offer: GpuOffer
    hourly_usd: float
    max_session_minutes: int
    existing: list[Pod]  # project pods from earlier sessions that still exist

    @property
    def max_cost_usd(self) -> float:
        return self.hourly_usd * self.max_session_minutes / 60


@dataclass
class PodSessionState:
    """What a session created, saved after every step so a crashed run can be torn down."""

    session_id: str
    pod_name: str = ""
    pod_id: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    data_center: str | None = None
    cuda_version: str | None = None
    vcpus: float | None = None
    memory_gb: float | None = None
    launched_at: datetime | None = None
    terminated_at: datetime | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), default=str, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> PodSessionState:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("launched_at", "terminated_at"):
            if raw.get(key):
                raw[key] = datetime.fromisoformat(raw[key])
        return cls(**raw)


def billed_seconds(launched: datetime, terminated: datetime) -> int:
    """Pods bill per second while they exist; whole seconds, rounded up."""
    return max(1, math.ceil((terminated - launched).total_seconds()))


def scenario_kind(path: Path) -> str:
    return str((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("kind", ""))


class NoDirectSshError(RuntimeError):
    """The pod runs but never got a public port for SSH."""


class PodSession:
    def __init__(
        self,
        settings: PodSettings,
        pods: Pods,
        *,
        session_id: str,
        repo: Path,
        state_dir: Path,
        clock: Clock,
        sleep: Callable[[float], None],
        remote_factory: Callable[[str, int, Path, Path], Remote],
        keygen: Callable[[Path, str], bytes],
        tracked_files: Callable[[Path], list[str]] | None = None,
    ) -> None:
        self.settings = settings
        self.pods = pods
        self.session_id = session_id
        self.repo = repo
        self.dir = state_dir / session_id
        self.clock = clock
        self._sleep = sleep
        self._remote_factory = remote_factory
        self._keygen = keygen
        self._tracked_files = tracked_files or git_tracked_files
        self.name = f"{settings.name_prefix}-gpu-{session_id.lower()}"
        self.state = PodSessionState(session_id=session_id, pod_name=self.name)

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    def _save(self) -> None:
        self.state.save(self.state_path)

    def _ours(self, pod: Pod) -> bool:
        return pod.name.startswith(f"{self.settings.name_prefix}-") and pod.status not in GONE

    # --- Preflight -----------------------------------------------------------------------

    def preflight(self) -> PodPreflight:
        """Checks that cost nothing and create nothing: price and stock, and old project pods."""
        s = self.settings
        offer = self.pods.gpu_offer(s.gpu_type, s.cloud, s.min_cuda_version)
        if offer.availability == "NONE":
            raise PreflightError(
                f"no {s.cloud} {offer.name} in stock for pods with CUDA >= {s.min_cuda_version}; "
                "try later, or another type with VMS_GPU_TYPE (GET /v2/catalog/gpus)"
            )
        existing = [p for p in self.pods.list_pods() if self._ours(p)]
        return PodPreflight(
            offer=offer,
            hourly_usd=s.hourly_usd(offer.hourly_usd),
            max_session_minutes=s.max_session_minutes,
            existing=existing,
        )

    # --- The session ---------------------------------------------------------------------

    def run(
        self,
        scenarios: list[str],
        confirm: Callable[[PodPreflight], bool],
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
            path = self.repo / scenario
            if not path.is_file():
                raise PreflightError(f"scenario not found: {scenario}")
            kind = scenario_kind(path)
            if kind not in POD_SCENARIO_KINDS:
                raise PreflightError(
                    f"{scenario} is kind {kind!r}; a pod runs only "
                    f"{sorted(POD_SCENARIO_KINDS)} scenarios (no Docker inside a pod, D37)"
                )
        pre = self.preflight()
        if not confirm(pre):
            log.info("session not confirmed; nothing was created")
            return 3

        started = self.clock.now()
        runs: list[ScenarioRun] = []
        collected: list[Path] = []
        remote_hw: Hardware | None = None
        timer: str | None = None
        error: str | None = None
        status = 0
        teardown = TeardownReport()
        try:
            remote = self._provision()
            timer = self._check_timer(remote)
            remote_hw = self._bootstrap(remote)
            self._sync(remote)
            remote.run(self._remote("setup"), self.settings.timeouts.setup_s)
            for scenario in scenarios:
                runs.append(self._run_scenario(remote, scenario))
            r = self.settings.remote
            collected = collect_new_files(
                remote,
                r.dir,
                r.collect,
                self.repo,
                results_dir,
                self.dir / "collect",
                self.settings.timeouts.collect_s,
            )
            if not all(run.ok for run in runs):
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
        cost = self._cost(pre.hourly_usd)
        note = f"GPU session {self.session_id} (Runpod {pre.offer.name}, {self.settings.cloud})"
        annotate_results(collected, git, pre.hourly_usd, note)
        record = self._record(
            pre, git, started, scenarios, runs, collected, remote_hw, timer, teardown, error, cost
        )
        path = write_result(results_dir, record)
        log.info("session results written", extra={"path": str(path), "cost_usd": cost})
        if not teardown.verified:
            return 2
        return status

    def _remote(self, *args: str) -> str:
        quoted = " ".join(shlex.quote(a) for a in args)
        return f"cd {shlex.quote(self.settings.remote.dir)} && bash {REMOTE_SCRIPT} {quoted}"

    def _spec(self, public_key: bytes) -> PodSpec:
        s = self.settings
        return PodSpec(
            name=self.name,
            image=s.image,
            gpu_type=s.gpu_type,
            gpu_count=s.gpu_count,
            cloud=s.cloud,
            min_cuda_version=s.min_cuda_version,
            container_disk_gb=s.container_disk_gb,
            ports=["22/tcp"],
            env={
                "PUBLIC_KEY": public_key.decode().strip(),
                "VMS_SESSION": self.session_id,
                "VMS_MAX_SECONDS": str(s.max_session_minutes * 60),
            },
            cmd=["bash", "-c", START_COMMAND],
            data_center_ids=s.data_center_ids,
        )

    def _provision(self) -> Remote:
        t = self.settings.timeouts
        key_path = self.dir / "id_ed25519"
        public_key = self._keygen(key_path, self.name)
        self._save()  # the pod name is known before the create, for teardown by name
        launched = self.clock.now()
        pod = self.pods.create_pod(self._spec(public_key))
        self.state.pod_id = pod.id
        self.state.launched_at = pod.created_at or launched
        self._save()
        log.info("pod created", extra={"pod_id": pod.id})
        running = self._wait_pod(lambda p: p is not None and p.status == "RUNNING", t.running_s)
        assert running is not None
        self.state.data_center, self.state.cuda_version = running.data_center, running.cuda_version
        self.state.vcpus, self.state.memory_gb = running.vcpus, running.memory_gb
        self._save()
        # Direct SSH needs a public port for 22/tcp. A host without one never gets it (E-063), so
        # give up early instead of paying for the whole running_s.
        try:
            ready = self._wait_pod(lambda p: p is not None and p.ssh is not None, t.ssh_direct_s)
        except TimeoutError as exc:
            last = self.pods.get_pod(running.id)
            ports = last.ports if last else ()
            raise NoDirectSshError(
                f"pod {running.id} in {running.data_center} is RUNNING but has no direct SSH "
                f"after {t.ssh_direct_s:.0f} s (ports {list(ports)}): the host likely has no "
                "public IP; try again (another host) or set data_center_ids"
            ) from exc
        assert ready is not None
        assert ready.ssh is not None
        self.state.ssh_host, self.state.ssh_port = ready.ssh.host, ready.ssh.port
        self.state.data_center, self.state.cuda_version = ready.data_center, ready.cuda_version
        self._save()
        remote = self._remote_factory(
            ready.ssh.host, ready.ssh.port, key_path, self.dir / "known_hosts"
        )
        remote.wait_ready(t.ssh_ready_s)
        return remote

    def _wait_pod(self, done: Callable[[Pod | None], bool], timeout_s: float) -> Pod | None:
        assert self.state.pod_id is not None
        deadline = self.clock.monotonic() + timeout_s
        while True:
            pod = self.pods.get_pod(self.state.pod_id)
            if done(pod):
                return pod
            if pod is not None and pod.status == "ERROR":
                raise RuntimeError(f"pod {pod.id} is in ERROR (GET /v2/pods/{pod.id}/logs)")
            if self.clock.monotonic() >= deadline:
                state = pod.status if pod else "gone"
                raise TimeoutError(f"pod {self.state.pod_id} still {state}")
            self._sleep(self.settings.timeouts.poll_s)

    def _check_timer(self, remote: Remote) -> str:
        """The pod's time limit must be armed with a key that can delete it."""
        out = remote.run(f"head -n 1 {TIMER_LOG} 2>/dev/null || true", 60, capture=True).strip()
        armed = out.startswith("armed") and "key=yes" in out
        if not armed and self.settings.require_self_terminate:
            raise RuntimeError(
                f"the pod's time limit is not armed ({out or 'no timer log'}): stopping before "
                "anything runs (require_self_terminate, configs/runpod.yaml)"
            )
        if not armed:
            log.warning("the pod's time limit is not armed", extra={"timer": out})
        return out

    def _bootstrap(self, remote: Remote) -> Hardware:
        for command in self.settings.host_setup:
            remote.run(command, self.settings.timeouts.setup_s)
        out = remote.run(f"python3 -c {shlex.quote(HW_PROBE)}", 120, capture=True)
        hw = Hardware.model_validate_json(out.strip().splitlines()[-1])
        if not hw.gpus:
            raise RuntimeError("nvidia-smi found no GPU in the pod")
        log.info("pod hardware", extra={"hardware": hw.model_dump()})
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

    # --- Teardown ------------------------------------------------------------------------

    def adopt_named(self) -> None:
        """Fill in the pod id from the account's pods (for `teardown` of a crashed run)."""
        if self.state.pod_id:
            return
        for pod in self.pods.list_pods():
            if pod.name == self.name and pod.status not in GONE:
                self.state.pod_id = pod.id
                return

    def teardown(self) -> TeardownReport:
        """Delete the session's pod (and any other pod with its name, e.g. from a create that
        timed out), then check that none remains."""
        report = TeardownReport()
        ids: list[str] = [self.state.pod_id] if self.state.pod_id else []
        try:
            ids += [
                p.id
                for p in self.pods.list_pods()
                if p.name == self.name and p.status not in GONE and p.id not in ids
            ]
        except Exception as exc:
            report.errors.append(f"list pods: {exc}")
        for pod_id in ids:
            try:
                self.pods.delete_pod(pod_id)
                self._wait_deleted(pod_id)
                if pod_id == self.state.pod_id:
                    self.state.terminated_at = self.clock.now()
            except Exception as exc:
                report.errors.append(f"delete {pod_id}: {exc}")
        self._save()
        try:
            report.leftovers = self.leftovers()
        except Exception as exc:
            report.errors.append(f"leftover check: {exc}")
        if report.verified:
            log.info("teardown verified: no pod of this session remains")
        else:
            log.error(
                "TEARDOWN NOT VERIFIED: a pod may still be billing",
                extra={"leftovers": report.leftovers, "errors": report.errors},
            )
        return report

    def _wait_deleted(self, pod_id: str) -> None:
        deadline = self.clock.monotonic() + self.settings.timeouts.terminate_s
        while True:
            pod = self.pods.get_pod(pod_id)
            if pod is None or pod.status in GONE:
                return
            if self.clock.monotonic() >= deadline:
                raise TimeoutError(f"pod {pod_id} still {pod.status}")
            self._sleep(self.settings.timeouts.poll_s)

    def leftovers(self) -> list[str]:
        """Pods named for this session, or recorded by it, that still exist."""
        found = {
            p.id for p in self.pods.list_pods() if p.name == self.name and p.status not in GONE
        }
        if self.state.pod_id:
            pod = self.pods.get_pod(self.state.pod_id)
            if pod is not None and pod.status not in GONE:
                found.add(pod.id)
        return sorted(found)

    # --- Results -------------------------------------------------------------------------

    def _billed(self) -> int:
        if self.state.launched_at is None:
            return 0
        end = self.state.terminated_at or self.clock.now()  # not confirmed: bill until now
        return billed_seconds(self.state.launched_at, end)

    def _cost(self, hourly: float) -> float:
        return round(hourly * self._billed() / 3600, 4)

    def _record(
        self,
        pre: PodPreflight,
        git: GitInfo,
        started: datetime,
        scenarios: list[str],
        runs: list[ScenarioRun],
        collected: list[Path],
        remote_hw: Hardware | None,
        timer: str | None,
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
            packages=package_versions(("pydantic",)),
            config=self.settings.model_dump(mode="json"),
            scenario={"scenarios": scenarios},
            metrics={
                "provider": "runpod",
                "cloud": self.settings.cloud,
                "gpu_type": pre.offer.id,
                "gpu_count": self.settings.gpu_count,
                "pod_id": s.pod_id,
                "pod_name": s.pod_name,
                "data_center": s.data_center,
                "host_cuda_version": s.cuda_version,
                "pod_vcpus": s.vcpus,
                "pod_memory_gb": s.memory_gb,
                "image": self.settings.image,
                "launched_at": s.launched_at.isoformat() if s.launched_at else None,
                "terminated_at": s.terminated_at.isoformat() if s.terminated_at else None,
                "billed_s": self._billed(),
                "gpu_hourly_usd": pre.offer.hourly_usd,
                "hourly_usd": round(pre.hourly_usd, 4),
                "time_limit": timer,
                "scenarios": [asdict(r) for r in runs],
                "results_files": [p.name for p in collected],
                "error": error,
                "teardown_verified": teardown.verified,
                "leftovers": teardown.leftovers,
                "teardown_errors": teardown.errors,
                "hardware_is": "pod" if remote_hw else "operator machine (no pod)",
            },
            cost_usd=cost,
            notes="cost estimated from the live catalog price x billed seconds; "
            "Runpod's billing history is the record",
        )
