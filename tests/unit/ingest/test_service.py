import asyncio
from datetime import UTC, datetime

import pytest

from tests.unit.ingest.conftest import (
    JPEG,
    NOW,
    FakeFrameRepository,
    FakeObjectStore,
    count,
)
from vms.common.ids import is_ulid
from vms.common.ingest_api import FrameMeta
from vms.common.metrics import Metrics
from vms.ingest.service import FrameIngestor, IngestError, Outcome, object_key

NOW_MS = int(NOW.timestamp() * 1000)


def meta(
    seq: int = 0, run_id: str = "run-1", camera_id: str = "cam-001", ts_ms: int = NOW_MS
) -> FrameMeta:
    return FrameMeta(camera_id=camera_id, run_id=run_id, seq=seq, ts_unix_ms=ts_ms)


def ingest(ingestor: FrameIngestor, m: FrameMeta, data: bytes = JPEG):  # type: ignore[no-untyped-def]
    return asyncio.run(ingestor.ingest(m, data))


def test_object_key_uses_the_capture_date_in_utc() -> None:
    ts = datetime(2026, 10, 9, 23, 30, tzinfo=UTC)
    assert object_key("cam-001", ts, "01ABC") == "frames/cam-001/2026-10-09/01ABC.jpg"


def test_new_frame_stores_object_then_row(
    ingestor: FrameIngestor, store: FakeObjectStore, repo: FakeFrameRepository, metrics: Metrics
) -> None:
    result = ingest(ingestor, meta())
    assert result.outcome is Outcome.CREATED
    assert not result.accepted.duplicate
    assert is_ulid(result.accepted.frame_id)
    row = repo.rows[("cam-001", "run-1", 0)]
    assert row.frame_id == result.accepted.frame_id
    assert row.ts == NOW
    assert row.size_bytes == len(JPEG)
    assert row.object_key == f"frames/cam-001/2026-10-09/{row.frame_id}.jpg"
    assert store.objects == {row.object_key: JPEG}
    assert count(metrics, "created") == 1


def test_resend_is_a_duplicate_with_the_original_id_and_no_extra_object(
    ingestor: FrameIngestor, store: FakeObjectStore, repo: FakeFrameRepository, metrics: Metrics
) -> None:
    first = ingest(ingestor, meta())
    second = ingest(ingestor, meta())
    assert second.outcome is Outcome.DUPLICATE
    assert second.accepted.duplicate
    assert second.accepted.frame_id == first.accepted.frame_id
    assert len(repo.rows) == 1
    assert list(store.objects) == [repo.rows[("cam-001", "run-1", 0)].object_key]
    assert count(metrics, "duplicate") == 1


def test_same_seq_in_another_run_or_camera_is_a_new_frame(
    ingestor: FrameIngestor, repo: FakeFrameRepository
) -> None:
    ingest(ingestor, meta(seq=0))
    ingest(ingestor, meta(seq=0, run_id="run-2"))
    ingest(ingestor, meta(seq=0, camera_id="cam-002"))
    assert len(repo.rows) == 3


def test_object_store_failure_stores_nothing(
    ingestor: FrameIngestor, store: FakeObjectStore, repo: FakeFrameRepository, metrics: Metrics
) -> None:
    store.fail_put = True
    with pytest.raises(IngestError) as err:
        ingest(ingestor, meta())
    assert (err.value.status, err.value.code) == (500, "object_store_unavailable")
    assert repo.rows == {}
    assert count(metrics, "error") == 1


def test_db_failure_after_object_write_removes_the_object(
    ingestor: FrameIngestor, store: FakeObjectStore, repo: FakeFrameRepository
) -> None:
    repo.fail_insert = True
    with pytest.raises(IngestError) as err:
        ingest(ingestor, meta())
    assert (err.value.status, err.value.code) == (500, "database_unavailable")
    assert store.objects == {}  # no orphan
    # The client's retry then succeeds normally.
    repo.fail_insert = False
    assert ingest(ingestor, meta()).outcome is Outcome.CREATED
    assert len(store.objects) == 1


def test_failed_cleanup_is_counted_as_an_orphan(
    ingestor: FrameIngestor, store: FakeObjectStore, repo: FakeFrameRepository, metrics: Metrics
) -> None:
    repo.fail_insert = True
    store.fail_delete = True
    with pytest.raises(IngestError):
        ingest(ingestor, meta())
    assert len(store.objects) == 1
    assert repo.rows == {}  # a row never points at a missing object; the reverse can happen
    assert metrics.registry.get_sample_value("vms_ingest_orphaned_objects_total") == 1


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 422, "empty_jpeg"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 10, 415, "not_jpeg"),
        (b"\xff\xd8\xff" + b"\x00" * 1000, 413, "jpeg_too_large"),
    ],
)
def test_invalid_images_are_rejected_before_any_write(
    ingestor: FrameIngestor,
    store: FakeObjectStore,
    metrics: Metrics,
    data: bytes,
    status: int,
    code: str,
) -> None:
    with pytest.raises(IngestError) as err:
        ingest(ingestor, meta(), data)
    assert (err.value.status, err.value.code) == (status, code)
    assert store.objects == {}
    assert count(metrics, "invalid") == 1


def test_capture_time_far_in_the_future_is_rejected(ingestor: FrameIngestor) -> None:
    with pytest.raises(IngestError) as err:
        ingest(ingestor, meta(ts_ms=NOW_MS + 301_000))
    assert err.value.code == "ts_in_future"
    # Small skew is normal (camera clocks drift) and accepted.
    assert ingest(ingestor, meta(seq=1, ts_ms=NOW_MS + 5_000)).outcome is Outcome.CREATED


def test_stage_latencies_are_recorded(ingestor: FrameIngestor, metrics: Metrics) -> None:
    ingest(ingestor, meta())
    for stage in ("object_put", "db_insert"):
        assert (
            metrics.registry.get_sample_value(
                "vms_ingest_stage_duration_seconds_count", {"stage": stage}
            )
            == 1
        )
