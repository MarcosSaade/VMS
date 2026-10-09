import pytest
from pydantic import ValidationError

from vms.common.ingest_api import FrameMeta


@pytest.mark.parametrize(
    "camera_id", ["cam-001", "CAM_7", "a", "lobby.east-2", "01M4GK6TJTHH0D2T3JW9KHWK28-r0"]
)
def test_safe_ids_are_accepted(camera_id: str) -> None:
    FrameMeta(camera_id=camera_id, run_id=camera_id, seq=0, ts_unix_ms=0)


@pytest.mark.parametrize(
    "camera_id", ["", ".", "..", "../x", "a/b", "a\\b", "a b", "a\nb", "é", "x" * 65, ".git"]
)
def test_ids_that_could_escape_an_object_key_are_rejected(camera_id: str) -> None:
    with pytest.raises(ValidationError):
        FrameMeta(camera_id=camera_id, run_id="r", seq=0, ts_unix_ms=0)
    with pytest.raises(ValidationError):
        FrameMeta(camera_id="c", run_id=camera_id, seq=0, ts_unix_ms=0)
