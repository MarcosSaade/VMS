import socket
from pathlib import Path

import pytest

from vms.common.ports import Published, conflicts, is_listening, main, message, own_ports, published

RAW = {
    "services": {
        "postgres": {"ports": ["127.0.0.1:${POSTGRES_PORT:-5432}:5432"]},
        "otel": {"ports": ["127.0.0.1:${OTEL_GRPC_PORT:-4317}:4317", "8888:8888"]},
        "worker": {"profiles": ["app"], "ports": ["127.0.0.1:${WORKER_PORT:-8090}:8090"]},
    }
}
# What `compose config --format json` resolves RAW to with POSTGRES_PORT=5433 and the app
# profile inactive (worker absent).
CONFIG = {
    "name": "vms",
    "services": {
        "postgres": {"ports": [{"host_ip": "127.0.0.1", "published": "5433", "target": 5432}]},
        "otel": {
            "ports": [
                {"host_ip": "127.0.0.1", "published": "4317", "target": 4317},
                {"published": "8888", "target": 8888},
                {"target": 9999},
            ]
        },
    },
}


def test_published_ports_carry_their_variable() -> None:
    assert published(CONFIG, RAW) == [
        Published("postgres", "127.0.0.1", 5433, "POSTGRES_PORT"),
        Published("otel", "127.0.0.1", 4317, "OTEL_GRPC_PORT"),
        Published("otel", "0.0.0.0", 8888, None),
    ]


def test_conflicts_skip_free_and_own_ports() -> None:
    ports = published(CONFIG, RAW)
    taken = {5433, 4317, 8888}
    listening = lambda _ip, port: port in taken  # noqa: E731
    assert [p.port for p in conflicts(ports, set(), listening)] == [5433, 4317, 8888]
    # the stack already running: its own published ports are not conflicts
    assert conflicts(ports, {5433, 4317, 8888}, listening) == []
    assert conflicts(ports, set(), lambda _ip, _port: False) == []


def test_own_ports_from_docker_ps() -> None:
    out = "127.0.0.1:5432->5432/tcp\n7333/tcp, 127.0.0.1:8333->8333/tcp, [::]:3000->3000/tcp\n\n"
    assert own_ports(out) == {5432, 8333, 3000}
    assert own_ports("127.0.0.1:4317-4318->4317-4318/tcp, 55679/tcp") == {4317, 4318}
    assert own_ports("") == set()


def test_message_names_the_variable() -> None:
    text = message([Published("postgres", "127.0.0.1", 5432, "POSTGRES_PORT")])
    assert "127.0.0.1:5432 (postgres): set POSTGRES_PORT in .env" in text
    assert "free it" in message([Published("otel", "0.0.0.0", 8888, None)])


def test_is_listening_against_a_real_socket() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert is_listening("127.0.0.1", port)
    assert not is_listening("127.0.0.1", port)


def test_main_fails_on_a_taken_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import json

    import yaml

    from vms.common import ports as mod

    compose = tmp_path / "compose.yml"
    compose.write_text(yaml.safe_dump(RAW))
    monkeypatch.setattr(mod, "_project_ports", lambda _project: set())
    monkeypatch.setattr(mod, "is_listening", lambda _ip, port: port == 5433)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(CONFIG)))
    assert main([str(compose)]) == 1
    assert "set POSTGRES_PORT in .env" in capsys.readouterr().err

    monkeypatch.setattr(mod, "is_listening", lambda _ip, _port: False)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(CONFIG)))
    assert main([str(compose)]) == 0
    assert main([]) == 2
