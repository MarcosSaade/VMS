"""Engine processes for the serving benchmark: start, wait until healthy, read, stop.

Each server is one child process in its own process group (so stopping it also stops the
workers vLLM forks), with its output in a log file whose tail goes into the results when a
start fails. Health, version and Prometheus reads are plain HTTP GETs with timeouts.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from vms.common.logs import get_logger
from vms.serving.scenario import Server
from vms.serving.settings import ServingSettings

log = get_logger(__name__)


class ServerStartError(RuntimeError):
    pass


class RunningServer(Protocol):
    @property
    def base_url(self) -> str: ...  # OpenAI-compatible root, ending in /v1

    def start(self) -> dict[str, Any]: ...  # info: version, startup seconds, model digest

    def metrics_text(self) -> str | None: ...  # Prometheus exposition, if the engine has one

    def stop(self) -> None: ...


def http_get(url: str, timeout_s: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, b""


def log_tail(path: Path, lines: int) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


class ProcessServer:
    """A server process started from a command, healthy when `health_path` answers 200."""

    def __init__(
        self,
        server: Server,
        command: list[str],
        root_url: str,
        health_path: str,
        log_path: Path,
        settings: ServingSettings,
        *,
        env: dict[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.server = server
        self.command = command
        self.root_url = root_url
        self.health_path = health_path
        self.log_path = log_path
        self.settings = settings
        self.env = env or {}
        self._clock = clock
        self._sleep = sleep
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"{self.root_url}/v1"

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        out = self.log_path.open("ab")
        log.info("server starting", extra={"server": self.server.name, "command": self.command})
        self._proc = subprocess.Popen(
            self.command,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**os.environ, **self.env, **self.server.env},  # the scenario's env wins
            start_new_session=True,
        )
        out.close()

    def _wait_healthy(self, timeout_s: float) -> None:
        deadline = self._clock() + timeout_s
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise ServerStartError(
                    f"{self.server.name} exited with {self._proc.returncode}:\n"
                    + log_tail(self.log_path, self.settings.log_tail_lines)
                )
            status, _ = http_get(self.root_url + self.health_path, self.settings.request_timeout_s)
            if status == 200:
                return
            if self._clock() >= deadline:
                raise ServerStartError(
                    f"{self.server.name} not healthy after {timeout_s:.0f} s:\n"
                    + log_tail(self.log_path, self.settings.log_tail_lines)
                )
            self._sleep(self.settings.poll_s)

    def _get_json(self, path: str) -> Any:
        status, body = http_get(self.root_url + path, self.settings.request_timeout_s)
        if status != 200:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return None

    def start(self) -> dict[str, Any]:
        t0 = self._clock()
        self._spawn()
        self._wait_healthy(self.server.start_timeout_s)
        return {"startup_s": self._clock() - t0, "version": self._get_json("/version")}

    def metrics_text(self) -> str | None:
        status, body = http_get(self.root_url + "/metrics", self.settings.request_timeout_s)
        return body.decode("utf-8", "replace") if status == 200 else None

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=self.settings.stop_timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=self.settings.stop_timeout_s)
        except ProcessLookupError:
            pass
        log.info("server stopped", extra={"server": self.server.name})


class OllamaServer(ProcessServer):
    """`ollama serve`, then `ollama pull` of the tag; the digest is recorded."""

    def start(self) -> dict[str, Any]:
        t0 = self._clock()
        self._spawn()
        self._wait_healthy(self.server.start_timeout_s)
        o = self.settings.ollama
        pull = subprocess.run(
            [o.bin, "pull", self.server.model],
            env={**os.environ, "OLLAMA_HOST": f"{o.host}:{o.port}"},
            capture_output=True,
            timeout=o.pull_timeout_s,
            check=False,
        )
        if pull.returncode != 0:
            raise ServerStartError(
                f"ollama pull {self.server.model} failed: {pull.stderr.decode()[-500:]}"
            )
        tags = self._get_json("/api/tags") or {}
        digest = next(
            (m.get("digest") for m in tags.get("models", []) if m.get("name") == self.server.model),
            None,
        )
        return {
            "startup_s": self._clock() - t0,
            "version": self._get_json("/api/version"),
            "digest": digest,
        }

    def metrics_text(self) -> str | None:
        return None  # Ollama exposes no Prometheus metrics


class ExternalServer:
    """An already running OpenAI-compatible endpoint (local smoke runs and tests)."""

    def __init__(self, server: Server) -> None:
        assert server.base_url is not None
        self._base = server.base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base

    def start(self) -> dict[str, Any]:
        return {"startup_s": 0.0, "version": None}

    def metrics_text(self) -> str | None:
        return None

    def stop(self) -> None:
        return None


def build_server(
    server: Server, settings: ServingSettings, logs_dir: Path, seed: int
) -> RunningServer:
    log_path = logs_dir / f"{server.name}.log"
    if server.engine == "external":
        return ExternalServer(server)
    if server.engine == "ollama":
        o = settings.ollama
        return OllamaServer(
            server,
            [o.bin, "serve"],
            o.root_url,
            "/api/version",
            log_path,
            settings,
            env={**o.env, "OLLAMA_HOST": f"{o.host}:{o.port}"},
        )
    v = settings.vllm
    assert server.revision is not None
    command = [
        v.bin,
        "serve",
        server.model,
        "--revision",
        server.revision,
        "--served-model-name",
        server.request_model,
        "--host",
        v.host,
        "--port",
        str(v.port),
        "--seed",
        str(seed),
        *server.args,
    ]
    return ProcessServer(server, command, v.root_url, "/health", log_path, settings, env=v.env)


def prometheus_counters(text: str | None, needle: str) -> dict[str, float]:
    """Sum of every sample whose metric name contains `needle`, by metric name (labels summed)."""
    out: dict[str, float] = {}
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if needle not in name:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def gpu_memory_used_mib() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]
