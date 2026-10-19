from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest

from tests.unit.cloud.conftest import CONFIGS, HW, T0, FakeRemote, git
from vms.bench.results import RunResult
from vms.cloud.pod_session import (
    KIND,
    START_COMMAND,
    PodSession,
    PodSessionState,
    billed_seconds,
)
from vms.cloud.pod_settings import SERVICE, PodSettings
from vms.cloud.remote import Remote
from vms.cloud.runpod import GpuOffer, Pod, PodSpec, SshEndpoint
from vms.cloud.session import PreflightError
from vms.common.clock import FakeClock
from vms.common.settings import load_settings

ARMED = "armed pod=pod-1 key=yes seconds=9000"


@dataclass
class FakePods:
    """Pods go PROVISIONING -> RUNNING (with SSH) on read, and are gone after a delete."""

    clock: FakeClock
    availability: str = "LOW"
    delete_fails: bool = False
    create_raises: bool = False  # the create times out after making the pod
    pods: dict[str, Pod] = field(default_factory=dict)
    created: list[PodSpec] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def gpu_offer(self, gpu_type: str, cloud: str, min_cuda_version: str) -> GpuOffer:
        return GpuOffer(gpu_type, "L4", 24, 0.59, self.availability, ["EU-RO-1"])

    def create_pod(self, spec: PodSpec) -> Pod:
        self.calls.append("create_pod")
        self.created.append(spec)
        pod = Pod(
            id=f"pod-{len(self.pods) + 1}",
            name=spec.name,
            status="PROVISIONING",
            created_at=self.clock.now(),
        )
        self.pods[pod.id] = pod
        if self.create_raises:
            raise TimeoutError("read timed out")
        return pod

    def get_pod(self, pod_id: str) -> Pod | None:
        self.clock.advance(1)
        pod = self.pods.get(pod_id)
        if pod is not None and pod.status == "PROVISIONING":
            self.pods[pod_id] = Pod(
                id=pod.id,
                name=pod.name,
                status="RUNNING",
                hourly_usd=0.59,
                data_center="EU-RO-1",
                cuda_version="13.2",
                ssh=SshEndpoint("203.0.113.9", 40122, "root"),
                created_at=pod.created_at,
            )
        return pod

    def list_pods(self) -> list[Pod]:
        return list(self.pods.values())

    def delete_pod(self, pod_id: str) -> None:
        self.calls.append(f"delete_pod {pod_id}")
        if self.delete_fails:
            raise RuntimeError("409 locked")
        self.pods.pop(pod_id, None)


@dataclass
class PodRemote(FakeRemote):
    timer: str = ARMED

    def run(self, command: str, timeout_s: float, *, capture: bool = False) -> str:
        if "vms-time-limit.log" in command:
            self.commands.append(command)
            return self.timer + "\n"
        return super().run(command, timeout_s, capture=capture)


@dataclass
class Harness:
    session: PodSession
    pods: FakePods
    remote: PodRemote
    clock: FakeClock
    repo: Path
    connects: list[tuple[str, int]]

    def run(self, scenarios: list[str] | None = None, confirm: bool = True, **kw: bool) -> int:
        return self.session.run(
            scenarios or ["bench/scenarios/worker-model-gpu-n.yaml"],
            lambda _pre: confirm,
            self.repo / "bench/results",
            **kw,
        )

    def record(self) -> RunResult:
        [path] = (self.repo / "bench/results").glob(f"*-session-{self.session.session_id}.json")
        return RunResult.model_validate_json(path.read_text())


@pytest.fixture
def settings() -> PodSettings:
    return load_settings(PodSettings, SERVICE, config_dir=CONFIGS)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "bench/scenarios").mkdir(parents=True)
    (root / "bench/results").mkdir(parents=True)
    (root / "bench/scenarios/worker-model-gpu-n.yaml").write_text("kind: model\n")
    (root / "bench/scenarios/gpu-smoke.yaml").write_text("kind: e2e\n")
    (root / "bench/results/already-committed.json").write_text("{}")
    (root / "data/clips/mot17-09-mini").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def make(settings: PodSettings, repo: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    def build(pods: FakePods | None = None, remote: PodRemote | None = None) -> Harness:
        clock = FakeClock(T0)
        pods = pods or FakePods(clock)
        pods.clock = clock
        remote = remote or PodRemote()
        connects: list[tuple[str, int]] = []

        def factory(host: str, port: int, key: Path, known: Path) -> Remote:
            connects.append((host, port))
            return remote

        def keygen(path: Path, comment: str) -> bytes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("private")
            return b"ssh-ed25519 AAAA test\n"

        session = PodSession(
            settings,
            pods,
            session_id="01SESSION",
            repo=repo,
            state_dir=tmp_path / "state",
            clock=clock,
            sleep=clock.advance,
            remote_factory=factory,
            keygen=keygen,
        )
        return Harness(session, pods, remote, clock, repo, connects)

    return build


def test_committed_config_prices_the_a40_from_the_catalog(settings: PodSettings) -> None:
    assert (settings.gpu_type, settings.cloud, settings.min_cuda_version) == (
        "NVIDIA A40",
        "SECURE",
        "13.0",
    )
    assert "@sha256:" in settings.image  # pinned by digest
    assert settings.require_self_terminate
    assert settings.hourly_usd(0.59) == pytest.approx(0.59 + 40 * 0.10 / 730)


def test_a_session_creates_runs_collects_deletes_and_verifies(make) -> None:  # type: ignore[no-untyped-def]
    h = make()
    assert h.run() == 0
    [spec] = h.pods.created
    assert spec.name == "vms-gpu-01session"
    assert spec.ports == ["22/tcp"]
    assert spec.env["PUBLIC_KEY"] == "ssh-ed25519 AAAA test"  # the session's own key
    assert spec.env["VMS_MAX_SECONDS"] == "9000"
    assert spec.cmd == ["bash", "-c", START_COMMAND]
    assert h.connects == [("203.0.113.9", 40122)]  # direct SSH, not the proxy
    cmds = h.remote.commands
    assert any("pod_remote.sh setup" in c for c in cmds)
    assert any("pod_remote.sh run bench/scenarios/worker-model-gpu-n.yaml" in c for c in cmds)
    assert "data/clips/mot17-09-mini" in h.remote.pushed
    assert ".git" in h.remote.pushed
    assert h.pods.pods == {}
    assert h.pods.calls[-1] == "delete_pod pod-1"
    rec = h.record()
    assert rec.kind == KIND
    assert rec.metrics["teardown_verified"] is True
    assert rec.metrics["pod_id"] == "pod-1"
    assert rec.metrics["data_center"] == "EU-RO-1"
    assert rec.metrics["host_cuda_version"] == "13.2"
    assert rec.metrics["time_limit"] == ARMED
    assert rec.hardware.gpus == HW.gpus
    assert rec.cost_usd is not None
    assert rec.cost_usd > 0
    collected = RunResult.model_validate_json(
        (h.repo / "bench/results/2026-10-13-gpu-smoke-run.json").read_text()
    )
    assert collected.git.sha is not None  # filled from the session
    assert "Runpod L4" in (collected.notes or "")


def test_only_model_scenarios_run_on_a_pod(make) -> None:  # type: ignore[no-untyped-def]
    h = make()
    with pytest.raises(PreflightError, match="kind 'e2e'"):
        h.run(["bench/scenarios/gpu-smoke.yaml"])
    assert h.pods.created == []


def test_out_of_stock_fails_preflight_and_creates_nothing(make) -> None:  # type: ignore[no-untyped-def]
    h = make(FakePods(FakeClock(T0), availability="NONE"))
    with pytest.raises(PreflightError, match="in stock"):
        h.run()
    assert h.pods.created == []


def test_not_confirmed_creates_nothing(make) -> None:  # type: ignore[no-untyped-def]
    h = make()
    assert h.run(confirm=False) == 3
    assert h.pods.created == []


def test_dirty_tree_is_refused(make) -> None:  # type: ignore[no-untyped-def]
    h = make()
    (h.repo / "bench/scenarios/worker-model-gpu-n.yaml").write_text("kind: model  # edited\n")
    with pytest.raises(PreflightError, match="uncommitted"):
        h.run()


def test_an_unarmed_time_limit_stops_the_session_before_anything_runs(make) -> None:  # type: ignore[no-untyped-def]
    h = make(remote=PodRemote(timer="armed pod=pod-1 key=no seconds=9000"))
    assert h.run() == 1
    assert not any("pod_remote.sh" in c for c in h.remote.commands)
    assert h.pods.pods == {}  # still deleted
    assert "not armed" in h.record().metrics["error"]


def test_a_failing_scenario_is_recorded_and_the_pod_is_still_deleted(make) -> None:  # type: ignore[no-untyped-def]
    h = make(remote=PodRemote(fail_on="run bench/scenarios"))
    assert h.run() == 1
    assert h.pods.pods == {}
    assert h.record().metrics["scenarios"][0]["ok"] is False


def test_ctrl_c_tears_down(make) -> None:  # type: ignore[no-untyped-def]
    h = make(remote=PodRemote(interrupt_on="pod_remote.sh setup"))
    assert h.run() == 130
    assert h.pods.pods == {}
    assert h.record().metrics["teardown_verified"] is True


def test_a_create_that_times_out_is_found_by_name_and_deleted(make) -> None:  # type: ignore[no-untyped-def]
    clock = FakeClock(T0)
    h = make(FakePods(clock, create_raises=True))
    assert h.run() == 1
    assert h.pods.pods == {}
    assert "delete_pod pod-1" in h.pods.calls


def test_a_pod_that_survives_teardown_fails_loudly(make) -> None:  # type: ignore[no-untyped-def]
    h = make(FakePods(FakeClock(T0), delete_fails=True))
    assert h.run() == 2
    rec = h.record()
    assert rec.metrics["teardown_verified"] is False
    assert rec.metrics["leftovers"] == ["pod-1"]


def test_teardown_of_a_crashed_session_from_saved_state(make) -> None:  # type: ignore[no-untyped-def]
    h = make()
    h.pods.pods["pod-7"] = Pod(id="pod-7", name="vms-gpu-01session", status="RUNNING")
    h.pods.pods["pod-8"] = Pod(id="pod-8", name="someone-elses-pod", status="RUNNING")
    h.session.adopt_named()
    assert h.session.state.pod_id == "pod-7"
    assert h.session.teardown().verified
    assert list(h.pods.pods) == ["pod-8"]  # only this session's pod is touched


def test_state_round_trips(tmp_path: Path) -> None:
    state = PodSessionState("01S", pod_name="vms-gpu-01s", pod_id="pod-1", launched_at=T0)
    state.save(tmp_path / "s.json")
    assert PodSessionState.load(tmp_path / "s.json") == state


def test_billing_is_per_second() -> None:
    assert billed_seconds(T0, T0 + timedelta(seconds=90.2)) == 91
    assert billed_seconds(T0, T0) == 1


def test_preflight_lists_old_project_pods(make) -> None:  # type: ignore[no-untyped-def]
    h = make()
    h.pods.pods["pod-9"] = Pod(id="pod-9", name="vms-gpu-old", status="EXITED")
    h.pods.pods["pod-10"] = Pod(id="pod-10", name="comfy", status="RUNNING")
    pre = h.session.preflight()
    assert [p.id for p in pre.existing] == ["pod-9"]
    assert pre.max_cost_usd == pytest.approx(pre.hourly_usd * 150 / 60)
