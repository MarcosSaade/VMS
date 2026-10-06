"""Fixtures for integration tests against the compose CPU stack (`make up`).

Endpoints and credentials come from the same `.env` the stack was started with, so port
overrides apply to both.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

HTTP_TIMEOUT_S = 5.0


class StackSettings(BaseSettings):
    """Host-side view of the local stack; defaults match docker-compose.yml."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "vms"
    postgres_user: str = "vms"
    postgres_password: str = "change-me"
    mlflow_db: str = "mlflow"

    s3_endpoint_url: str = "http://localhost:8333"
    s3_bucket: str = "vms-frames"
    s3_access_key_id: str = "vms-local"
    s3_secret_access_key: str = "change-me-s3"
    mlflow_artifacts_bucket: str = "vms-mlflow"

    redis_url: str = "redis://localhost:6379/0"

    otel_grpc_port: int = 4317
    prometheus_port: int = 9090
    tempo_port: int = 3200
    grafana_port: int = 3000
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "change-me-grafana"
    mlflow_port: int = 5000

    @property
    def postgres_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} user={self.postgres_user} "
            f"password={self.postgres_password} connect_timeout=5"
        )

    def url(self, port: int, path: str = "") -> str:
        return f"http://localhost:{port}{path}"


@pytest.fixture(scope="session")
def stack() -> StackSettings:
    settings = StackSettings()
    try:
        urllib.request.urlopen(settings.url(settings.grafana_port, "/api/health"), timeout=2)
    except OSError as exc:
        pytest.fail(f"compose stack is not reachable ({exc}); run `make up` first")
    return settings


def get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.load(response)


def poll(check: Callable[[], Any], *, timeout_s: float, interval_s: float = 0.5) -> Any:
    """Call `check` until it returns a truthy value or the deadline passes.

    Only for state that becomes visible asynchronously inside an external system (for example
    a span flushed by the collector into Tempo); there is no event to wait on instead.
    """
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except OSError as exc:  # includes URLError and HTTPError
            last_error = exc
        time.sleep(interval_s)
    raise AssertionError(f"condition not met within {timeout_s}s (last error: {last_error})")
