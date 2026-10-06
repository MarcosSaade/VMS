import io
import urllib.error
from pathlib import Path

import pytest

from tests.unit.datasets.conftest import FakeTransport, sha
from vms.datasets.http import (
    ChecksumError,
    DownloadError,
    RemoteFile,
    RetryPolicy,
    download_file,
)

URL = "https://example.test/file.bin"
DATA = bytes(range(256)) * 400  # 100 KiB


def test_retry_succeeds_after_transient_errors() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(attempts=3, base_s=1.0, max_s=10.0, sleep=sleeps.append)
    calls = iter([OSError("reset"), OSError("reset"), "ok"])

    def flaky() -> str:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    assert policy.run("x", flaky) == "ok"
    assert len(sleeps) == 2
    assert 0 <= sleeps[0] <= 1.0  # full jitter, exponential cap
    assert 0 <= sleeps[1] <= 2.0


def test_retry_gives_up_after_bounded_attempts() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(attempts=3, sleep=sleeps.append)

    def broken() -> None:
        raise TimeoutError("slow")

    with pytest.raises(DownloadError, match="after 3 attempts"):
        policy.run("x", broken)
    assert len(sleeps) == 2


def test_retry_does_not_retry_client_errors() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(attempts=5, sleep=sleeps.append)

    def missing() -> None:
        raise urllib.error.HTTPError(URL, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    with pytest.raises(DownloadError, match="HTTP 404"):
        policy.run("x", missing)
    assert sleeps == []


def test_retry_is_deterministic_with_a_seeded_rng() -> None:
    def delays() -> list[float]:
        sleeps: list[float] = []
        policy = RetryPolicy(attempts=4, sleep=sleeps.append)
        with pytest.raises(DownloadError):
            policy.run("x", lambda: (_ for _ in ()).throw(OSError("x")))
        return sleeps

    assert delays() == delays()


def test_download_verifies_and_is_idempotent(tmp_path: Path, retry: RetryPolicy) -> None:
    transport = FakeTransport({URL: DATA})
    dest = tmp_path / "raw" / "file.bin"
    download_file(transport, retry, URL, dest, sha("sha256", DATA), len(DATA))
    assert dest.read_bytes() == DATA
    download_file(transport, retry, URL, dest, sha("sha256", DATA), len(DATA))
    assert len(transport.requests) == 1  # second call reuses the verified file


@pytest.mark.parametrize("algo", ["md5", "sha1", "sha256", "sha512"])
def test_download_supports_published_checksum_algorithms(
    tmp_path: Path, retry: RetryPolicy, algo: str
) -> None:
    transport = FakeTransport({URL: DATA})
    download_file(transport, retry, URL, tmp_path / "f", sha(algo, DATA), len(DATA))


def test_checksum_mismatch_leaves_nothing_behind(tmp_path: Path, retry: RetryPolicy) -> None:
    transport = FakeTransport({URL: DATA})
    dest = tmp_path / "file.bin"
    with pytest.raises(ChecksumError):
        download_file(transport, retry, URL, dest, sha("sha256", b"other"), len(DATA))
    assert list(tmp_path.iterdir()) == []


def test_corrupt_cached_file_is_downloaded_again(tmp_path: Path, retry: RetryPolicy) -> None:
    transport = FakeTransport({URL: DATA})
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"corrupt")
    download_file(transport, retry, URL, dest, sha("sha256", DATA), len(DATA))
    assert dest.read_bytes() == DATA


def test_short_and_failed_reads_are_retried(tmp_path: Path, retry: RetryPolicy) -> None:
    transport = FakeTransport({URL: DATA})
    transport.truncate_next = 1
    transport.fail_next = 1
    dest = tmp_path / "file.bin"
    download_file(transport, retry, URL, dest, sha("sha256", DATA), len(DATA))
    assert dest.read_bytes() == DATA
    assert len(transport.requests) == 3


def test_remote_file_reads_arbitrary_ranges(retry: RetryPolicy) -> None:
    transport = FakeTransport({URL: DATA})
    remote = RemoteFile(transport, retry, URL, len(DATA))
    reader = io.BufferedReader(remote, buffer_size=1024)
    reader.seek(1000)
    assert reader.read(10) == DATA[1000:1010]
    reader.seek(-5, io.SEEK_END)
    assert reader.read() == DATA[-5:]


def test_prefetched_span_is_served_locally(tmp_path: Path, retry: RetryPolicy) -> None:
    transport = FakeTransport({URL: DATA})
    remote = RemoteFile(transport, retry, URL, len(DATA))
    remote.prefetch(10_000, 60_000, tmp_path)
    before = len(transport.requests)
    for offset in range(10_000, 60_000, 7_000):
        remote.seek(offset)
        assert remote.read(100) == DATA[offset : offset + 100]
    assert len(transport.requests) == before
    remote.seek(80_000)
    assert remote.read(10) == DATA[80_000:80_010]  # outside the span: one range request
    assert len(transport.requests) == before + 1
    remote.close()
    assert list(tmp_path.iterdir()) == []  # the spool file is anonymous
