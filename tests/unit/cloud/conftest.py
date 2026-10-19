"""Fakes for GPU sessions: an in-memory AWS (`FakeCloud`) and instance (`FakeRemote`)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vms.bench.results import GitInfo, Hardware, RunResult
from vms.cloud.aws import BudgetSpec, Identity, Image, Instance, LaunchDeniedError, LaunchSpec
from vms.cloud.remote import RemoteError
from vms.cloud.session import GpuSession
from vms.cloud.settings import SERVICE, CloudSettings
from vms.common.clock import FakeClock
from vms.common.settings import load_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "configs"
T0 = datetime(2026, 10, 13, 12, tzinfo=UTC)
ACCOUNT = "123456789012"
REGION = "us-east-2"
HW = Hardware(cpu="AMD EPYC", logical_cpus=4, ram_bytes=16, platform="linux", gpus=["NVIDIA L4"])


def arn(kind: str, ident: str) -> str:
    return f"arn:aws:ec2:{REGION}:{ACCOUNT}:{kind}/{ident}"


@dataclass
class FakeCloud:
    """Instances move pending -> running and shutting-down -> terminated on describe."""

    clock: FakeClock
    quota: float = 4.0
    deny_launch: bool = False
    terminate_fails: bool = False
    sg_busy_polls: int = 1  # deletes refused (DependencyViolation) before one succeeds
    stale_tagged: list[str] = field(default_factory=list)  # tagging API lag: gone, still listed
    tags: dict[str, dict[str, str]] = field(default_factory=dict)  # arn -> tags, existing only
    instances: dict[str, str] = field(default_factory=dict)  # id -> state
    calls: list[str] = field(default_factory=list)
    launched: list[LaunchSpec] = field(default_factory=list)
    budgets: list[BudgetSpec] = field(default_factory=list)

    def identity(self) -> Identity:
        return Identity(account=ACCOUNT, arn="arn:aws:sts::x", region=REGION)

    def vcpu_quota(self, quota_code: str) -> float:
        return self.quota

    def instance_vcpus(self, instance_type: str) -> int:
        return 4

    def resolve_image(self, parameter: str) -> Image:
        return Image(id="ami-1", name="DLAMI", root_device="/dev/sda1")

    def check_launch(self, spec: LaunchSpec) -> None:
        if self.deny_launch:
            raise LaunchDeniedError("explicit deny in a service control policy")

    def import_key_pair(self, name: str, public_key: bytes, tags: dict[str, str]) -> str:
        self.calls.append("import_key_pair")
        self.tags[arn("key-pair", "key-1")] = tags
        return "key-1"

    def create_security_group(self, name: str, ssh_cidr: str, tags: dict[str, str]) -> str:
        self.calls.append(f"create_security_group {ssh_cidr}")
        self.tags[arn("security-group", "sg-1")] = tags
        return "sg-1"

    def run_instance(self, spec: LaunchSpec) -> str:
        self.calls.append("run_instance")
        self.launched.append(spec)
        self.instances["i-1"] = "pending"
        self.tags[arn("instance", "i-1")] = spec.tags
        self.tags[arn("volume", "vol-1")] = spec.tags
        return "i-1"

    def describe_instance(self, instance_id: str) -> Instance | None:
        state = self.instances.get(instance_id)
        if state is None:
            return None
        nxt = {"pending": "running", "shutting-down": "terminated"}.get(state, state)
        self.instances[instance_id] = nxt
        self.clock.advance(1)
        return Instance(id=instance_id, state=state, public_ip="203.0.113.7", launch_time=T0)

    def terminate_instance(self, instance_id: str) -> None:
        self.calls.append("terminate_instance")
        if self.terminate_fails:
            raise RuntimeError("AccessDenied")
        if instance_id in self.instances:
            self.instances[instance_id] = "shutting-down"

    def delete_security_group(self, group_id: str) -> bool:
        self.calls.append("delete_security_group")
        if self.sg_busy_polls > 0:
            self.sg_busy_polls -= 1
            return False
        self.tags.pop(arn("security-group", group_id), None)
        return True

    def delete_key_pair(self, key_pair_id: str) -> None:
        self.calls.append("delete_key_pair")
        self.tags.pop(arn("key-pair", key_pair_id), None)

    def tagged_arns(self, tags: dict[str, str]) -> list[str]:
        live = [a for a, t in self.tags.items() if tags.items() <= t.items()]
        return sorted([*live, *self.stale_tagged])

    def exists(self, resource_arn: str) -> bool:
        kind, _, ident = resource_arn.split(":")[5].partition("/")
        if kind == "instance":
            return self.instances.get(ident, "terminated") != "terminated"
        if kind == "volume":  # deleted on termination
            return self.instances.get("i-1", "terminated") != "terminated" and (
                resource_arn in self.tags
            )
        return resource_arn in self.tags

    def budget_exists(self, account: str, name: str) -> bool:
        return any(b.name == name for b in self.budgets)

    def create_budget(self, account: str, spec: BudgetSpec) -> None:
        self.budgets.append(spec)


def results_file(name: str, *, sha: str | None = None, seconds: float = 360) -> RunResult:
    start = T0
    return RunResult(
        run_id=f"run-{name}",
        name=name,
        kind="bench.e2e",
        started_at=start,
        ended_at=datetime.fromtimestamp(start.timestamp() + seconds, tz=UTC),
        seed=1,
        git=GitInfo(sha=sha, dirty=None if sha is None else False),
        hardware=HW,
        packages={},
        config={},
        metrics={},
    )


@dataclass
class FakeRemote:
    """Runs nothing; `pull` produces one new results file and one already in the repo."""

    fail_on: str | None = None  # substring of a command that fails
    interrupt_on: str | None = None
    commands: list[str] = field(default_factory=list)
    pushed: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    repo_results: str = "already-committed.json"

    def wait_ready(self, timeout_s: float) -> None:
        self.commands.append("wait_ready")

    def run(self, command: str, timeout_s: float, *, capture: bool = False) -> str:
        self.commands.append(command)
        if self.interrupt_on and self.interrupt_on in command:
            raise KeyboardInterrupt
        if self.fail_on and self.fail_on in command:
            raise RemoteError(f"exit 1: {command}")
        if "python3 -c" in command:
            return "some noise\n" + HW.model_dump_json() + "\n"
        return ""

    def push(self, local_root: Path, paths: list[str], remote_dir: str, timeout_s: float) -> None:
        self.pushed = paths

    def pull(self, remote_dir: str, local_dir: Path, timeout_s: float) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        self.pulled.append(remote_dir)
        if remote_dir.endswith("models/finetune"):
            (local_dir / "ft.pt").write_bytes(b"\x80weights")
            return
        if remote_dir.endswith("eval/results"):
            new = results_file("eval-run", sha=None)
            (local_dir / "2026-10-13-eval-run.json").write_text(new.model_dump_json())
            return
        new = results_file("gpu-smoke", sha=None)
        (local_dir / "2026-10-13-gpu-smoke-run.json").write_text(new.model_dump_json())
        (local_dir / self.repo_results).write_text("{}")


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def settings() -> CloudSettings:
    return load_settings(CloudSettings, SERVICE, config_dir=CONFIGS)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clean git checkout with one scenario and a results file already committed."""
    root = tmp_path / "repo"
    (root / "bench/scenarios").mkdir(parents=True)
    (root / "bench/results").mkdir(parents=True)
    (root / "bench/scenarios/gpu-smoke.yaml").write_text("kind: e2e\n")
    (root / "bench/results/already-committed.json").write_text("{}")
    (root / "data/clips/mot17-09-mini").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


@dataclass
class Harness:
    session: GpuSession
    cloud: FakeCloud
    remote: FakeRemote
    clock: FakeClock
    repo: Path

    def run(self, confirm: bool = True, **kwargs: bool) -> int:
        return self.session.run(
            ["bench/scenarios/gpu-smoke.yaml"],
            lambda _pre: confirm,
            self.repo / "bench/results",
            **kwargs,
        )


@pytest.fixture
def make_harness(settings: CloudSettings, repo: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    def make(cloud: FakeCloud | None = None, remote: FakeRemote | None = None) -> Harness:
        clock = FakeClock(T0)
        cloud = cloud or FakeCloud(clock)
        cloud.clock = clock
        remote = remote or FakeRemote()

        def keygen(path: Path, comment: str) -> bytes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("private")
            return b"ssh-ed25519 AAAA test"

        session = GpuSession(
            settings,
            cloud,
            session_id="01SESSION",
            repo=repo,
            state_dir=tmp_path / "state",
            clock=clock,
            sleep=clock.advance,
            remote_factory=lambda host, key, known: remote,
            keygen=keygen,
            ssh_cidr=lambda: "198.51.100.1/32",
        )
        return Harness(session, cloud, remote, clock, repo)

    return make
