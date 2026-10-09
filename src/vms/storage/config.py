"""Connection settings for Postgres and the object store, shared by every service.

They read the same variables the compose stack is started with (`POSTGRES_*`, `S3_*` in `.env`;
see `.env.example`), so one `.env` configures both the stack and the services. They are kept out
of the per-service YAML because they hold secrets and differ per environment, not per service.
Pool sizes and timeouts, which are tuning rather than identity, live in each service's YAML.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder for the local object store, which ignores it. AWS runs read AWS_REGION.
DEFAULT_REGION = "us-east-1"


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = "vms"
    user: str = "vms"
    password: SecretStr = SecretStr("change-me")

    def conninfo(self, *, connect_timeout_s: float = 5.0, application_name: str = "vms") -> str:
        """A libpq connection string (password included: never log it)."""
        return (
            f"host={self.host} port={self.port} dbname={self.db} user={self.user} "
            f"password={self.password.get_secret_value()} "
            f"connect_timeout={max(1, round(connect_timeout_s))} "
            f"application_name={application_name}"
        )

    def sqlalchemy_url(self) -> str:
        """For Alembic, which drives migrations through SQLAlchemy and psycopg 3."""
        from sqlalchemy.engine import URL

        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.db,
        ).render_as_string(hide_password=False)


class ObjectStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="S3_", env_file=".env", extra="ignore")

    # Empty means AWS S3 itself; locally, the SeaweedFS S3 gateway.
    endpoint_url: str | None = "http://localhost:8333"
    bucket: str = "vms-frames"
    access_key_id: SecretStr | None = SecretStr("vms-local")
    secret_access_key: SecretStr | None = SecretStr("change-me-s3")
    # SeaweedFS ignores the region; on AWS it must be the project's selected Region.
    region: str = Field(
        default=DEFAULT_REGION, validation_alias=AliasChoices("S3_REGION", "AWS_REGION")
    )

    @field_validator("endpoint_url", "access_key_id", "secret_access_key", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        """Empty keys mean "use the default AWS credential chain" (instance role, profile)."""
        return None if value == "" else value

    @field_validator("region", mode="before")
    @classmethod
    def _empty_region_is_default(cls, value: object) -> object:
        return DEFAULT_REGION if value == "" else value
