"""Fixtures for integration tests against the compose CPU stack (`make up`).

Endpoints and credentials come from the same `.env` the stack was started with, so port
overrides apply to both.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import boto3
import psycopg
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from botocore.config import Config
from psycopg import sql
from pydantic_settings import BaseSettings, SettingsConfigDict

from vms.common.ids import new_ulid
from vms.storage.config import PostgresSettings

HTTP_TIMEOUT_S = 5.0
REPO_ROOT = Path(__file__).resolve().parents[2]


class StackSettings(BaseSettings):
    """Host-side view of the local stack; defaults match docker-compose.yml."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "vms"
    postgres_user: str = "vms"
    postgres_password: str = "change-me"

    s3_endpoint_url: str = "http://localhost:8333"
    s3_bucket: str = "vms-frames"
    s3_access_key_id: str = "vms-local"
    s3_secret_access_key: str = "change-me-s3"

    otel_grpc_port: int = 4317
    prometheus_port: int = 9090
    tempo_port: int = 3200
    grafana_port: int = 3000
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "change-me-grafana"

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


@pytest.fixture
def s3(stack: StackSettings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=stack.s3_endpoint_url,
        aws_access_key_id=stack.s3_access_key_id,
        aws_secret_access_key=stack.s3_secret_access_key,
        region_name="us-east-1",  # signing Region for the local endpoint only
        config=Config(connect_timeout=HTTP_TIMEOUT_S, read_timeout=HTTP_TIMEOUT_S),
    )


@pytest.fixture
def fresh_db(stack: StackSettings) -> Iterator[str]:
    """A throwaway database migrated to head; dropped afterwards. Yields its name.

    Tests that write frames use it instead of the dev database, so they start empty and leave
    nothing behind.
    """
    name = f"vms_it_{new_ulid().lower()}"
    admin = f"{stack.postgres_dsn} dbname=postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        command.upgrade(alembic_config(stack, name), "head")
        yield name
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def alembic_config(stack: StackSettings, db: str) -> AlembicConfig:
    url = PostgresSettings(
        _env_file=None,  # type: ignore[call-arg]
        host=stack.postgres_host,
        port=stack.postgres_port,
        user=stack.postgres_user,
        password=stack.postgres_password,
        db=db,
    ).sqlalchemy_url()
    config = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config
