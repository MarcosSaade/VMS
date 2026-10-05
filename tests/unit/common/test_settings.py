from pathlib import Path

import pytest
from pydantic import ValidationError

from vms.common.settings import ServiceSettings, deep_merge, load_settings


class IngestSettings(ServiceSettings):
    port: int = 8000
    max_frame_bytes: int = 500_000


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    write(tmp_path / "common.yaml", "log:\n  level: info\ntelemetry:\n  enabled: true\n")
    write(tmp_path / "ingest.yaml", "port: 9000\ntelemetry:\n  sample_ratio: 0.5\n")
    write(tmp_path / "ingest.cloud.yaml", "port: 9100\n")
    return tmp_path


def test_defaults_without_yaml_layers(tmp_path: Path) -> None:
    write(tmp_path / "ingest.yaml", "")
    settings = load_settings(IngestSettings, "ingest", config_dir=tmp_path)
    assert settings.service == "ingest"
    assert settings.env == "local"
    assert settings.port == 8000
    assert settings.telemetry.enabled is False
    assert settings.tracking.backends == ["noop"]
    assert settings.tracking.wandb.mode is None


def test_yaml_layers_deep_merge(config_dir: Path) -> None:
    settings = load_settings(IngestSettings, "ingest", config_dir=config_dir)
    assert settings.port == 9000
    assert settings.telemetry.enabled is True  # from common.yaml
    assert settings.telemetry.sample_ratio == 0.5  # from ingest.yaml, same section
    assert settings.log.level == "INFO"


def test_env_overlay_selected_by_vms_env(config_dir: Path) -> None:
    settings = load_settings(
        IngestSettings, "ingest", config_dir=config_dir, environ={"VMS_ENV": "cloud"}
    )
    assert settings.env == "cloud"
    assert settings.port == 9100


def test_environment_overrides_yaml(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VMS_PORT", "9500")
    monkeypatch.setenv("VMS_LOG__LEVEL", "debug")
    monkeypatch.setenv("VMS_TRACKING__BACKENDS", '["mlflow", "wandb"]')
    monkeypatch.setenv("VMS_TELEMETRY__SAMPLE_RATIO", "0.1")
    settings = load_settings(IngestSettings, "ingest", config_dir=config_dir)
    assert settings.port == 9500
    assert settings.log.level == "DEBUG"
    assert settings.tracking.backends == ["mlflow", "wandb"]
    assert settings.telemetry.sample_ratio == 0.1
    assert settings.telemetry.enabled is True  # untouched siblings keep their YAML value


def test_keyword_overrides_beat_environment(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VMS_PORT", "9500")
    settings = load_settings(IngestSettings, "ingest", config_dir=config_dir, port=1234)
    assert settings.port == 1234


def test_dotenv_file_is_read_when_given(config_dir: Path, tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    write(env_file, "VMS_PORT=7777\n")
    settings = load_settings(IngestSettings, "ingest", config_dir=config_dir, env_file=env_file)
    assert settings.port == 7777


def test_dotenv_ignores_unrelated_keys_but_reads_vms_ones(config_dir: Path, tmp_path: Path) -> None:
    # The shared .env also holds POSTGRES_*, WANDB_* etc. for other consumers.
    env_file = tmp_path / "test.env"
    write(env_file, "POSTGRES_HOST=db\nWANDB_MODE=offline\nVMS_PORT=7777\nVMS_LOG__LEVEL=debug\n")
    settings = load_settings(IngestSettings, "ingest", config_dir=config_dir, env_file=env_file)
    assert (settings.port, settings.log.level) == (7777, "DEBUG")


def test_dotenv_keys_for_other_services_are_ignored(tmp_path: Path) -> None:
    # Same behaviour as the process environment: a shared .env may set another service's field.
    write(tmp_path / "query.yaml", "")
    env_file = tmp_path / "test.env"
    write(env_file, "VMS_PORT=7777\nVMS_SEED=5\n")
    settings = load_settings(ServiceSettings, "query", config_dir=tmp_path, env_file=env_file)
    assert settings.seed == 5


def test_repo_env_example_loads(config_dir: Path) -> None:
    env_example = Path(__file__).parents[3] / ".env.example"
    settings = load_settings(IngestSettings, "ingest", config_dir=config_dir, env_file=env_example)
    assert settings.tracking.backends == ["noop"]


def test_missing_service_config_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest"):
        load_settings(IngestSettings, "ingest", config_dir=tmp_path)


@pytest.mark.parametrize(
    "yaml_text",
    [
        "prot: 9000\n",  # typo in a top-level key
        "telemetry:\n  enabld: true\n",  # typo in a nested key
        "telemetry:\n  sample_ratio: 1.5\n",  # out of range
        "tracking:\n  backends: [mlfow]\n",  # unknown backend
        "log:\n  level: LOUD\n",
    ],
)
def test_invalid_config_fails_fast(tmp_path: Path, yaml_text: str) -> None:
    write(tmp_path / "ingest.yaml", yaml_text)
    with pytest.raises(ValidationError):
        load_settings(IngestSettings, "ingest", config_dir=tmp_path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    write(tmp_path / "ingest.yaml", "- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(IngestSettings, "ingest", config_dir=tmp_path)


def test_loads_do_not_leak_between_calls(tmp_path: Path) -> None:
    write(tmp_path / "ingest.yaml", "port: 1111\n")
    write(tmp_path / "other.yaml", "")
    first = load_settings(IngestSettings, "ingest", config_dir=tmp_path)
    second = load_settings(IngestSettings, "other", config_dir=tmp_path)
    assert (first.port, second.port) == (1111, 8000)
    assert IngestSettings().port == 8000


def test_deep_merge_replaces_lists_and_scalars() -> None:
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    merged = deep_merge(base, {"a": {"c": [3]}, "d": {"e": 2}})
    assert merged == {"a": {"b": 1, "c": [3]}, "d": {"e": 2}}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 1}  # inputs untouched


def test_repo_common_config_is_valid(tmp_path: Path) -> None:
    repo_common = Path(__file__).parents[3] / "configs" / "common.yaml"
    write(tmp_path / "common.yaml", repo_common.read_text(encoding="utf-8"))
    write(tmp_path / "svc.yaml", "")
    settings = load_settings(ServiceSettings, "svc", config_dir=tmp_path)
    assert settings.tracking.backends == ["noop"]
