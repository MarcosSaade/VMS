import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.unit.cloud.conftest import (
    ACCOUNT,
    REGION,
    T0,
    FakeCloud,
    FakeRemote,
    Harness,
    arn,
    git,
)
from vms.bench.results import RunResult
from vms.cloud.session import KIND, PreflightError, SessionState, billed_seconds
from vms.cloud.settings import CloudSettings
from vms.common.clock import FakeClock


def session_record(h: Harness) -> RunResult:
    [path] = (h.repo / "bench/results").glob(f"*-session-{h.session.session_id}.json")
    return RunResult.model_validate_json(path.read_text())


def test_hourly_cost_from_the_committed_config(settings: CloudSettings) -> None:
    assert settings.machine == "gpu"
    assert settings.target.instance_type == "g6.xlarge"
    expected = 0.8048 + 100 * 0.08 / 730 + 0.005
    assert settings.hourly_usd() == pytest.approx(expected)
    with pytest.raises(ValueError, match="no price"):
        settings.model_copy(update={"instance_type": "p5.48xlarge"}).hourly_usd()


def test_the_cpu_machine_and_an_instance_type_override(settings: CloudSettings) -> None:
    cpu = settings.model_copy(update={"machine": "cpu"})
    assert cpu.target.accelerator == "cpu"
    assert cpu.target.vcpu_quota_code == "L-1216C47A"  # standard, not G and VT
    assert cpu.hourly_usd() == pytest.approx(0.357 + 60 * 0.08 / 730 + 0.005)
    bigger = cpu.model_copy(update={"instance_type": "c7i.4xlarge"})
    assert bigger.target.instance_type == "c7i.4xlarge"
    assert bigger.target.ami_parameter == cpu.target.ami_parameter  # the rest is the machine's
    with pytest.raises(ValueError, match="unknown machine"):
        _ = settings.model_copy(update={"machine": "tpu"}).target


def test_the_machine_is_selected_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.cloud.conftest import CONFIGS
    from vms.cloud.settings import SERVICE
    from vms.common.settings import load_settings

    monkeypatch.setenv("VMS_MACHINE", "cpu")
    monkeypatch.setenv("VMS_INSTANCE_TYPE", "c7i.4xlarge")
    s = load_settings(CloudSettings, SERVICE, config_dir=CONFIGS)
    assert (s.target.accelerator, s.target.instance_type) == ("cpu", "c7i.4xlarge")


@pytest.mark.parametrize(
    ("seconds", "billed"), [(0, 60), (59.2, 60), (60, 60), (60.1, 61), (3600, 3600)]
)
def test_billed_seconds_per_second_with_a_minute_minimum(seconds: float, billed: int) -> None:
    assert billed_seconds(T0, T0 + timedelta(seconds=seconds)) == billed


def test_a_session_runs_scenarios_tears_down_and_records_its_cost(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    assert h.run() == 0

    # Provisioned once, SSH only from the operator, torn down completely.
    assert h.cloud.calls[:3] == [
        "import_key_pair",
        "create_security_group 198.51.100.1/32",
        "run_instance",
    ]
    assert h.cloud.instances == {"i-1": "terminated"}
    assert [a for a in h.cloud.tagged_arns({"session": "01SESSION"}) if h.cloud.exists(a)] == []
    spec = h.cloud.launched[0]
    assert spec.tags == {"project": "vms", "session": "01SESSION", "Name": "vms-gpu-01session"}
    assert "shutdown -h +150" in spec.user_data
    assert spec.root_volume_gb == 100

    # Code and clips synced; setup, then each scenario, through the remote script.
    assert ".git" in h.remote.pushed
    assert "bench/scenarios/gpu-smoke.yaml" in h.remote.pushed
    assert "data/clips/mot17-09-mini" in h.remote.pushed  # present locally
    assert "data/clips/commons-roncesvalles" not in h.remote.pushed  # absent: fetched there
    scripted = [c for c in h.remote.commands if "gpu_remote.sh" in c]
    assert scripted == [
        "cd vms && VMS_ACCELERATOR=gpu bash scripts/gpu_remote.sh setup",
        "cd vms && VMS_ACCELERATOR=gpu bash scripts/gpu_remote.sh run "
        "bench/scenarios/gpu-smoke.yaml",
    ]

    record = session_record(h)
    assert record.kind == KIND
    assert record.hardware.gpus == ["NVIDIA L4"]  # the instance's, not this machine's
    assert record.git.sha is not None
    assert record.git.dirty is False
    m = record.metrics
    assert m["teardown_verified"] is True
    assert m["leftovers"] == []
    assert m["teardown_errors"] == []
    assert m["image"] == {"id": "ami-1", "name": "DLAMI"}
    assert m["results_files"] == [
        "2026-10-13-gpu-smoke-run.json",
        "2026-10-13-eval-run.json",
        "ft.pt",
    ]
    billed = m["billed_s"]
    assert billed == billed_seconds(T0, datetime.fromisoformat(m["terminated_at"]))
    assert record.cost_usd == pytest.approx(
        h.session.settings.hourly_usd() * billed / 3600, abs=1e-4
    )


def test_collected_results_get_their_cost_and_the_session_git_sha(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    h.run()
    results = h.repo / "bench/results"
    smoke = RunResult.model_validate_json((results / "2026-10-13-gpu-smoke-run.json").read_text())
    assert smoke.git.sha == session_record(h).git.sha
    assert smoke.cost_usd == pytest.approx(h.session.settings.hourly_usd() * 360 / 3600, abs=1e-4)
    assert "GPU session 01SESSION (g6.xlarge)" in smoke.notes
    # Eval results and weights land at their own repo paths; only JSON results are annotated.
    assert h.remote.pulled == ["vms/bench/results", "vms/eval/results", "vms/models/finetune"]
    assert any("mkdir -p eval/results models/finetune" in c for c in h.remote.commands)
    ev = RunResult.model_validate_json(
        (h.repo / "eval/results/2026-10-13-eval-run.json").read_text()
    )
    assert ev.cost_usd > 0
    assert (h.repo / "models/finetune/ft.pt").read_bytes() == b"\x80weights"
    # A file that came from the repo is not overwritten or claimed.
    assert (results / "already-committed.json").read_text() == "{}"


def test_setup_failure_still_tears_down(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(remote=FakeRemote(fail_on="setup"))
    assert h.run() == 1
    assert h.cloud.instances == {"i-1": "terminated"}
    assert "delete_key_pair" in h.cloud.calls
    m = session_record(h).metrics
    assert "RemoteError" in m["error"]
    assert m["teardown_verified"] is True
    assert not any("run bench" in c for c in h.remote.commands)


def test_a_failed_scenario_is_recorded_and_the_session_continues(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(remote=FakeRemote(fail_on="run bench"))
    assert h.run() == 1
    m = session_record(h).metrics
    assert m["scenarios"][0]["ok"] is False
    assert m["results_files"] == [
        "2026-10-13-gpu-smoke-run.json",
        "2026-10-13-eval-run.json",
        "ft.pt",
    ]  # still collected
    assert m["teardown_verified"] is True


def test_ctrl_c_mid_scenario_tears_down(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(remote=FakeRemote(interrupt_on="run bench"))
    assert h.run() == 130
    assert h.cloud.instances == {"i-1": "terminated"}
    assert session_record(h).metrics["error"] == "interrupted"


def test_a_resource_that_survives_teardown_fails_loudly(make_harness, caplog) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(cloud=FakeCloud(FakeClock(T0), terminate_fails=True))
    assert h.run() == 2
    m = session_record(h).metrics
    assert arn("instance", "i-1") in m["leftovers"]
    assert any("terminate i-1" in e for e in m["teardown_errors"])
    assert m["teardown_verified"] is False
    assert "TEARDOWN NOT VERIFIED" in caplog.text
    # Billing could not be confirmed to stop: cost runs to the end of the session, not 0.
    assert session_record(h).cost_usd > 0


def test_tagging_api_lag_on_a_terminated_instance_is_not_a_leftover(make_harness) -> None:  # type: ignore[no-untyped-def]
    cloud = FakeCloud(FakeClock(T0), stale_tagged=[arn("instance", "i-0old")])
    h: Harness = make_harness(cloud=cloud)
    assert h.run() == 0
    assert session_record(h).metrics["leftovers"] == []


def test_security_group_deletion_waits_for_the_network_interface(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(cloud=FakeCloud(FakeClock(T0), sg_busy_polls=3))
    assert h.run() == 0
    assert h.cloud.calls.count("delete_security_group") == 4


def test_a_security_group_that_never_frees_is_a_leftover(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(cloud=FakeCloud(FakeClock(T0), sg_busy_polls=10**6))
    assert h.run() == 2
    assert session_record(h).metrics["leftovers"] == [arn("security-group", "sg-1")]


def test_no_quota_fails_preflight_before_creating_anything(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(cloud=FakeCloud(FakeClock(T0), quota=0))
    with pytest.raises(PreflightError, match="quota"):
        h.run()
    assert h.cloud.calls == []


def test_a_denied_dry_run_fails_preflight(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(cloud=FakeCloud(FakeClock(T0), deny_launch=True))
    with pytest.raises(PreflightError, match="service control policy"):
        h.run()
    assert h.cloud.calls == []


def test_preflight_reports_leftovers_from_earlier_sessions(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    h.cloud.tags[arn("security-group", "sg-old")] = {"project": "vms", "session": "old"}
    pre = h.session.preflight()
    assert pre.existing == [arn("security-group", "sg-old")]
    assert pre.max_cost_usd == pytest.approx(pre.hourly_usd * 150 / 60)


def test_not_confirmed_creates_nothing(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    assert h.run(confirm=False) == 3
    assert h.cloud.calls == []
    assert not list((h.repo / "bench/results").glob("*gpu-session*"))


def test_uncommitted_changes_refuse_to_start_unless_allowed(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    (h.repo / "bench/scenarios/gpu-smoke.yaml").write_text("kind: e2e  # edited\n")
    with pytest.raises(PreflightError, match="uncommitted"):
        h.run()
    assert h.cloud.calls == []
    assert h.run(allow_dirty=True) == 0
    assert session_record(h).git.dirty is True


def test_a_missing_scenario_fails_before_launch(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    with pytest.raises(PreflightError, match="not found"):
        h.session.run(["bench/scenarios/nope.yaml"], lambda _p: True, h.repo / "bench/results")
    assert h.cloud.calls == []


def test_region_mismatch_fails_preflight(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    h.session.settings = h.session.settings.model_copy(update={"region": "eu-west-1"})
    with pytest.raises(PreflightError, match="Region"):
        h.session.preflight()


def test_state_is_saved_after_each_resource_and_round_trips(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    h.run()
    state = SessionState.load(h.session.state_path)
    assert (state.instance_id, state.security_group_id, state.key_pair_id) == (
        "i-1",
        "sg-1",
        "key-1",
    )
    assert state.launched_at == T0
    assert state.terminated_at is not None
    assert state.arns() == [
        arn("instance", "i-1"),
        arn("security-group", "sg-1"),
        arn("key-pair", "key-1"),
    ]
    raw = json.loads(h.session.state_path.read_text())
    assert raw["account"] == ACCOUNT
    assert raw["region"] == REGION


def test_teardown_of_a_crashed_session_finds_resources_by_tag(make_harness) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    tags = {"project": "vms", "session": "01SESSION"}
    h.cloud.instances["i-9"] = "running"
    h.cloud.tags[arn("instance", "i-9")] = tags
    h.cloud.tags[arn("security-group", "sg-9")] = tags
    h.cloud.tags[arn("key-pair", "key-9")] = tags
    h.cloud.tags[arn("security-group", "sg-other")] = {"project": "vms", "session": "other"}
    h.session.adopt_tagged()
    report = h.session.teardown()
    assert report.verified
    assert h.cloud.instances["i-9"] == "terminated"
    remaining = [a for a in h.cloud.tagged_arns({"project": "vms"}) if h.cloud.exists(a)]
    assert remaining == [arn("security-group", "sg-other")]


def test_bootstrap_requires_a_gpu(make_harness) -> None:  # type: ignore[no-untyped-def]
    class NoGpu(FakeRemote):
        def run(self, command: str, timeout_s: float, *, capture: bool = False) -> str:
            out = super().run(command, timeout_s, capture=capture)
            return out.replace('["NVIDIA L4"]', "[]")

    h: Harness = make_harness(remote=NoGpu())
    assert h.run() == 1
    assert "no GPU" in session_record(h).metrics["error"]
    assert h.cloud.instances == {"i-1": "terminated"}


def test_a_cpu_session_prepares_the_host_and_needs_no_gpu(  # type: ignore[no-untyped-def]
    make_harness, settings: CloudSettings
) -> None:
    class CpuOnly(FakeRemote):
        def run(self, command: str, timeout_s: float, *, capture: bool = False) -> str:
            out = super().run(command, timeout_s, capture=capture)
            return out.replace('["NVIDIA L4"]', "[]")

    h: Harness = make_harness(remote=CpuOnly())
    h.session.settings = settings.model_copy(update={"machine": "cpu"})
    assert h.run() == 0
    host_setup = h.session.settings.target.host_setup
    assert host_setup
    assert h.remote.commands[1 : 1 + len(host_setup)] == host_setup
    assert any("docker.io" in c for c in host_setup)  # Docker before the docker group
    usermod = next(i for i, c in enumerate(h.remote.commands) if "usermod" in c)
    assert usermod == 1 + len(host_setup)
    scripted = [c for c in h.remote.commands if "gpu_remote.sh" in c]
    assert scripted[0] == "cd vms && VMS_ACCELERATOR=cpu bash scripts/gpu_remote.sh setup"
    record = session_record(h)
    assert record.hardware.gpus == []
    assert record.metrics["machine"] == "cpu"
    assert record.metrics["instance_type"] == "c7i.2xlarge"
    assert record.cost_usd == pytest.approx(
        h.session.settings.hourly_usd() * record.metrics["billed_s"] / 3600, abs=1e-4
    )
    smoke = RunResult.model_validate_json(
        (h.repo / "bench/results/2026-10-13-gpu-smoke-run.json").read_text()
    )
    assert "CPU session 01SESSION (c7i.2xlarge)" in smoke.notes


def test_a_cpu_preflight_checks_the_standard_quota(make_harness, settings: CloudSettings) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness(cloud=FakeCloud(FakeClock(T0), quota=2.0))
    h.session.settings = settings.model_copy(update={"machine": "cpu"})
    with pytest.raises(PreflightError, match=r"on-demand standard .*L-1216C47A"):
        h.session.preflight()


def test_not_a_git_checkout_refuses(make_harness, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    h: Harness = make_harness()
    h.session.repo = tmp_path / "not-a-repo"
    h.session.repo.mkdir()
    with pytest.raises(PreflightError, match="git"):
        h.run()
    git(h.repo, "status")  # the fixture repo itself is untouched
