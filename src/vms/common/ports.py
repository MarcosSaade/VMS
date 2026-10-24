"""Preflight for `make up`: are the host ports the stack publishes free?

When a default port is already taken on the host (a local Postgres on 5432 is common), Docker
fails part way through `compose up`. A second `make up` can then report the container healthy
without its port published, so the migrations reach the host's own server and fail with a
misleading "password authentication failed" (found by the fresh-clone test, task 11.4). This
check runs first and names the variable to change. Ports published by this compose project's
own containers count as free: `make up` on a running stack must still pass.

    docker compose ... config --format json | python -m vms.common.ports COMPOSE_FILE

The resolved config (stdin) gives the ports of the services the active profiles start; the raw
compose file gives the `*_PORT` variable behind each one.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONNECT_TIMEOUT_S = 0.5
DOCKER_TIMEOUT_S = 10.0
# "127.0.0.1:${POSTGRES_PORT:-5432}:5432" -> POSTGRES_PORT
_PORT_VARIABLE = re.compile(r"^(?:[^:$]+:)?\$\{(\w+)")
# `docker ps` Ports column: "127.0.0.1:5432->5432/tcp, 127.0.0.1:4317-4318->4317-4318/tcp"
_HOST_PORTS = re.compile(r":(\d+)(?:-(\d+))?->")


@dataclass(frozen=True)
class Published:
    service: str
    host_ip: str
    port: int
    variable: str | None


def published(config: Mapping[str, Any], raw: Mapping[str, Any]) -> list[Published]:
    """Host ports in the resolved compose config, each with its variable from the raw file."""
    out: list[Published] = []
    raw_services = raw.get("services") or {}
    for name, service in (config.get("services") or {}).items():
        raw_ports = (raw_services.get(name) or {}).get("ports") or []
        for i, port in enumerate(service.get("ports") or []):
            if not port.get("published"):
                continue
            variable = None
            if i < len(raw_ports) and isinstance(raw_ports[i], str):
                match = _PORT_VARIABLE.match(raw_ports[i])
                variable = match.group(1) if match else None
            host_ip = port.get("host_ip") or "0.0.0.0"
            out.append(Published(name, host_ip, int(port["published"]), variable))
    return out


def own_ports(docker_ps_ports: str) -> set[int]:
    """Host ports published by containers, from `docker ps --format '{{.Ports}}'` output."""
    ports: set[int] = set()
    for first, last in _HOST_PORTS.findall(docker_ps_ports):
        ports.update(range(int(first), int(last or first) + 1))
    return ports


def is_listening(host_ip: str, port: int, timeout_s: float = CONNECT_TIMEOUT_S) -> bool:
    host = "127.0.0.1" if host_ip in ("0.0.0.0", "") else host_ip
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def conflicts(
    ports: Sequence[Published], own: set[int], listening: Callable[[str, int], bool]
) -> list[Published]:
    """Ports some other process on the host already listens on."""
    return [p for p in ports if p.port not in own and listening(p.host_ip, p.port)]


def message(taken: Sequence[Published]) -> str:
    lines = ["Host ports the stack needs are in use by another process:"]
    for p in taken:
        fix = f"set {p.variable} in .env" if p.variable else "free it"
        lines.append(f"  {p.host_ip}:{p.port} ({p.service}): {fix}")
    lines.append(
        "Pick free ports, update the matching URLs (.env.example names them), then make up again."
    )
    return "\n".join(lines)


def _project_ports(project: str) -> set[int]:
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.Ports}}",
            ],
            capture_output=True,
            text=True,
            timeout=DOCKER_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return set()  # compose itself reports a missing or unreachable Docker
    return own_ports(result.stdout)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: compose config --format json | python -m vms.common.ports FILE\n")
        return 2
    config = json.load(sys.stdin)
    raw = yaml.safe_load(Path(args[0]).read_text(encoding="utf-8")) or {}
    taken = conflicts(published(config, raw), _project_ports(config.get("name", "")), is_listening)
    if taken:
        sys.stderr.write(message(taken) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
