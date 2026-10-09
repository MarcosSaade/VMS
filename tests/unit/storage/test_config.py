from pathlib import Path

import pytest

from vms.storage.config import DEFAULT_REGION, ObjectStoreSettings, PostgresSettings

ENV_EXAMPLE = Path(__file__).resolve().parents[3] / ".env.example"


def test_env_example_parses_including_inline_comments() -> None:
    pg = PostgresSettings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]
    assert (pg.host, pg.port, pg.db, pg.user) == ("localhost", 5432, "vms", "vms")
    s3 = ObjectStoreSettings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]
    assert s3.endpoint_url == "http://localhost:8333"
    assert s3.bucket == "vms-frames"
    assert s3.region == DEFAULT_REGION  # AWS_REGION is empty there


def test_empty_endpoint_and_keys_mean_aws_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_ENDPOINT_URL", "")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    s3 = ObjectStoreSettings(_env_file=None)  # type: ignore[call-arg]
    assert s3.endpoint_url is None
    assert s3.access_key_id is None
    assert s3.region == "eu-west-1"


def test_conninfo_and_url_carry_the_password_but_repr_does_not() -> None:
    pg = PostgresSettings(_env_file=None, password="s3cr3t", db="x")  # type: ignore[call-arg]
    assert "password=s3cr3t" in pg.conninfo()
    assert "dbname=x" in pg.conninfo()
    assert pg.sqlalchemy_url() == "postgresql+psycopg://vms:s3cr3t@localhost:5432/x"
    assert "s3cr3t" not in repr(pg)
