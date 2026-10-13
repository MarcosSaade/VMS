"""The GPU instance seen from this machine: SSH commands and rsync, behind `Remote`.

`SshRemote` shells out to `ssh` and `rsync` with a per-session key and known_hosts file, a
connect timeout and keepalives, and a hard timeout on every call. Long commands stream their
output to this terminal so a session's progress is visible.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol


class RemoteError(Exception):
    """A remote command or transfer failed or timed out."""


class Remote(Protocol):
    def wait_ready(self, timeout_s: float) -> None: ...
    def run(self, command: str, timeout_s: float, *, capture: bool = False) -> str:
        """Run a shell command on the instance; raises `RemoteError` on failure."""
        ...

    def push(self, local_root: Path, paths: list[str], remote_dir: str, timeout_s: float) -> None:
        """Copy `paths` (relative to `local_root`, files or directories) under `remote_dir`."""
        ...

    def pull(self, remote_dir: str, local_dir: Path, timeout_s: float) -> None: ...


def generate_key(path: Path, comment: str) -> bytes:
    """A fresh ed25519 key pair at `path` (private) and `path.pub`; returns the public key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path.with_suffix(".pub").read_bytes()


class SshRemote:
    def __init__(
        self,
        host: str,
        user: str,
        key: Path,
        known_hosts: Path,
        connect_timeout_s: int,
        poll_s: float = 5.0,
    ) -> None:
        self._target = f"{user}@{host}"
        self._ssh = [
            "ssh",
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            # A new instance's host key is unknown; accept it once, then pin it for the session.
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            f"ConnectTimeout={connect_timeout_s}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ]
        self._poll_s = poll_s

    def _call(self, cmd: list[str], timeout_s: float, capture: bool) -> str:
        try:
            done = subprocess.run(
                cmd,
                timeout=timeout_s,
                check=False,
                text=True,
                stdout=subprocess.PIPE if capture else sys.stderr,
                stderr=subprocess.PIPE if capture else sys.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"timed out after {timeout_s:.0f} s: {' '.join(cmd[-2:])}") from exc
        if done.returncode != 0:
            detail = (done.stderr or "").strip()[-2000:] if capture else ""
            raise RemoteError(f"exit {done.returncode}: {' '.join(cmd[-2:])} {detail}".strip())
        return (done.stdout or "") if capture else ""

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.run("true", timeout_s=30, capture=True)
                return
            except RemoteError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(self._poll_s)

    def run(self, command: str, timeout_s: float, *, capture: bool = False) -> str:
        return self._call([*self._ssh, self._target, command], timeout_s, capture)

    def _rsync(self) -> list[str]:
        return ["rsync", "-az", "--partial", "-e", " ".join(self._ssh)]

    def push(self, local_root: Path, paths: list[str], remote_dir: str, timeout_s: float) -> None:
        files = "\n".join(paths) + "\n"
        cmd = [
            *self._rsync(),
            "--recursive",
            "--files-from=-",
            f"{local_root}/",
            f"{self._target}:{remote_dir}/",
        ]
        try:
            done = subprocess.run(
                cmd, input=files, text=True, timeout=timeout_s, check=False, capture_output=True
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"push timed out after {timeout_s:.0f} s") from exc
        if done.returncode != 0:
            raise RemoteError(f"push failed (exit {done.returncode}): {done.stderr[-2000:]}")

    def pull(self, remote_dir: str, local_dir: Path, timeout_s: float) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = [*self._rsync(), f"{self._target}:{remote_dir}/", f"{local_dir}/"]
        self._call(cmd, timeout_s, capture=True)
