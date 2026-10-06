"""Typed, layered service configuration.

Each service defines a subclass of `ServiceSettings` and loads it with `load_settings`. Values are
resolved from these layers, highest priority first:

1. Keyword arguments passed to `load_settings` (tests, scripts).
2. Environment variables `VMS_<FIELD>`, with `__` for nesting: `VMS_LOG__LEVEL=DEBUG`,
   `VMS_TELEMETRY__SAMPLE_RATIO=0.1`.
3. An optional dotenv file (`.env`), same names as 2.
4. YAML files in `configs/`, deep-merged in order: `common.yaml` (optional, shared by every
   service), `<service>.yaml` (required), `<service>.<env>.yaml` (optional; `env` comes from
   `VMS_ENV`, default `local`).
5. Defaults declared on the model.

Why: all configuration lives in versioned YAML (reproducibility), deployments override only what
differs through the environment (twelve-factor), and pydantic validates everything at startup, so
a typo in a config fails fast with a clear error instead of misbehaving at runtime. The fully
resolved settings are what benchmark results record.

Third-party variables such as `OTEL_EXPORTER_OTLP_ENDPOINT` are read by those libraries
themselves whenever the matching field here is left unset (`None`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_DIR = Path("configs")
COMMON_CONFIG = "common"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class TelemetrySettings(_Section):
    """OpenTelemetry tracing. Disabled means the API's no-op tracer: spans cost ~nothing."""

    enabled: bool = False
    # None defers to OTEL_EXPORTER_OTLP_ENDPOINT, then the exporter default (localhost:4317).
    otlp_endpoint: str | None = None
    otlp_insecure: bool = True
    export_timeout_s: float = Field(default=10.0, gt=0)
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class ServiceSettings(BaseSettings):
    """Settings shared by every service. Subclass and add service-specific fields."""

    model_config = SettingsConfigDict(
        env_prefix="VMS_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
    )

    service: str = "vms"
    env: str = "local"
    seed: int = 1234
    log: LoggingSettings = Field(default_factory=LoggingSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    # Merged YAML layers; set per load by `load_settings` on a throwaway subclass.
    _yaml_data: ClassVar[dict[str, Any]] = {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = _DictSource(settings_cls, cls._yaml_data)
        dotenv_source = _DeclaredFieldsOnly(settings_cls, dotenv_settings)
        return (init_settings, env_settings, dotenv_source, yaml_source)


class _DictSource(PydanticBaseSettingsSource):
    """A settings source backed by an already-merged dict (the YAML layers)."""

    def __init__(self, settings_cls: type[BaseSettings], data: Mapping[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = dict(data)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class _DeclaredFieldsOnly(PydanticBaseSettingsSource):
    """Keeps only dotenv keys that map to a declared field, like the process-environment source.

    One `.env` is shared by every service and by other tools (POSTGRES_*, OTEL_*, another
    service's VMS_PORT), so unknown keys are ignored rather than rejected by `extra="forbid"`.
    Typos are still caught in the per-service YAML files, which are validated strictly.
    """

    def __init__(self, settings_cls: type[BaseSettings], inner: PydanticBaseSettingsSource) -> None:
        super().__init__(settings_cls)
        self._inner = inner

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._inner.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        fields = self.settings_cls.model_fields
        return {key: value for key, value in self._inner().items() if key in fields}


def config_files(service: str, env: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> list[Path]:
    """The YAML layers for a service, lowest priority first. Optional layers may not exist."""
    return [
        config_dir / f"{COMMON_CONFIG}.yaml",
        config_dir / f"{service}.yaml",
        config_dir / f"{service}.{env}.yaml",
    ]


def load_settings[S: ServiceSettings](
    settings_cls: type[S],
    service: str,
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    **overrides: Any,
) -> S:
    """Load `settings_cls` for `service` from the YAML layers, environment and overrides.

    `environ` is only used to pick the env overlay (`VMS_ENV`); field values from the
    environment are read by pydantic-settings from the real process environment.
    """
    environ = os.environ if environ is None else environ
    env = str(overrides.get("env") or environ.get("VMS_ENV") or "local")

    common, base, overlay = config_files(service, env, config_dir)
    if not base.is_file():
        raise FileNotFoundError(f"missing config for service {service!r}: {base}")
    data: dict[str, Any] = {"service": service, "env": env}
    for path in (common, base, overlay):
        if path.is_file():
            data = deep_merge(data, _read_yaml(path))

    bound = type(settings_cls.__name__, (settings_cls,), {"_yaml_data": data})
    settings = bound(_env_file=env_file, **overrides)
    assert isinstance(settings, settings_cls)
    return settings


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; values in `override` win. Lists are replaced, not merged."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded
