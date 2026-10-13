import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.cloud.conftest import REPO_ROOT
from vms.cloud.remote import RemoteError, SshRemote


class Recorder:
    def __init__(self, returncode: int = 0, stdout: str = "", timeout: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.returncode, self.stdout, self.timeout = returncode, stdout, timeout

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"cmd": cmd, **kwargs})
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, "boom")


@pytest.fixture
def remote(tmp_path: Path) -> SshRemote:
    return SshRemote("203.0.113.7", "ubuntu", tmp_path / "key", tmp_path / "known", 10)


def test_ssh_pins_the_session_key_and_host_key(
    remote: SshRemote, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rec = Recorder(stdout="ok\n")
    monkeypatch.setattr(subprocess, "run", rec)
    assert remote.run("nvidia-smi", 30, capture=True) == "ok\n"
    cmd = rec.calls[0]["cmd"]
    assert cmd[:3] == ["ssh", "-i", str(tmp_path / "key")]
    for option in (
        "BatchMode=yes",
        "StrictHostKeyChecking=accept-new",
        f"UserKnownHostsFile={tmp_path / 'known'}",
        "ConnectTimeout=10",
    ):
        assert option in cmd
    assert cmd[-2:] == ["ubuntu@203.0.113.7", "nvidia-smi"]
    assert rec.calls[0]["timeout"] == 30


def test_failures_and_timeouts_raise(remote: SshRemote, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Recorder(returncode=3))
    with pytest.raises(RemoteError, match="exit 3"):
        remote.run("false", 5, capture=True)
    monkeypatch.setattr(subprocess, "run", Recorder(timeout=True))
    with pytest.raises(RemoteError, match="timed out"):
        remote.run("sleep 99", 5)


def test_push_sends_the_file_list_over_ssh(
    remote: SshRemote, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rec = Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    remote.push(tmp_path, ["a.py", ".git"], "vms", 60)
    call = rec.calls[0]
    assert call["cmd"][0] == "rsync"
    assert "--files-from=-" in call["cmd"]
    assert call["cmd"][-2:] == [f"{tmp_path}/", "ubuntu@203.0.113.7:vms/"]
    assert call["input"] == "a.py\n.git\n"
    assert call["cmd"][call["cmd"].index("-e") + 1].startswith("ssh -i ")


def test_gpu_override_runs_the_worker_on_cuda_with_pinned_images() -> None:
    override = yaml.safe_load((REPO_ROOT / "deploy/compose/docker-compose.gpu.yml").read_text())
    worker = override["services"]["worker"]
    assert worker["environment"]["VMS_DETECTOR__DEVICE"] == "cuda"
    assert worker["build"]["dockerfile"] == "deploy/docker/Dockerfile.gpu"
    assert "@sha256:" in override["services"]["dcgm-exporter"]["image"]


def test_scripts_are_valid_bash() -> None:
    for script in ("scripts/gpu_session.sh", "scripts/gpu_remote.sh"):
        subprocess.run(["bash", "-n", str(REPO_ROOT / script)], check=True, timeout=10)
