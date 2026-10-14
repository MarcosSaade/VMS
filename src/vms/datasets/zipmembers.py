"""Read scattered members of a remote zip, one range request each, in parallel.

`vms.datasets.clips` reads a contiguous frame sequence as one span. An eval subset is the
opposite case: a few hundred members spread over a whole archive (500 of COCO val2017's 5,000
images), where one span would download most of the archive. Here the central directory is read
once, then each member is fetched with its own range request (local header, name, extra field
and data) and decompressed locally. CRC32 is checked for every member; the caller still pins
the bytes with a sha256.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from vms.datasets.http import (
    ChecksumError,
    DownloadError,
    RemoteFile,
    RetryPolicy,
    Transport,
    read_range,
)

LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # 30 bytes
LOCAL_SIGNATURE = b"PK\x03\x04"
# Room for the local header's own name and extra field, which may differ from the central
# directory's (zip64 or timestamp extras); one request covers both.
HEADER_SLACK = 30 + 1024


def remote_zip_index(
    transport: Transport, retry: RetryPolicy, url: str, expected_bytes: int
) -> dict[str, zipfile.ZipInfo]:
    """Central directory of a remote zip by member name. The archive size pins its version."""
    size = transport.size(url)
    if size != expected_bytes:
        raise ChecksumError(f"{url}: archive is {size} bytes, spec pins {expected_bytes}")
    remote = RemoteFile(transport, retry, url, size)
    try:
        with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as archive:
            return {info.filename: info for info in archive.infolist() if not info.is_dir()}
    finally:
        remote.close()


def member_from_span(info: zipfile.ZipInfo, span: bytes) -> bytes:
    """Decompress one member from bytes starting at its local header; checks CRC32."""
    if len(span) < LOCAL_HEADER.size:
        raise DownloadError(f"{info.filename}: short local header")
    fields = LOCAL_HEADER.unpack_from(span)
    if fields[0] != LOCAL_SIGNATURE:
        raise DownloadError(f"{info.filename}: bad local header signature")
    name_len, extra_len = fields[9], fields[10]
    start = LOCAL_HEADER.size + name_len + extra_len
    raw = span[start : start + info.compress_size]
    if len(raw) != info.compress_size:
        raise DownloadError(f"{info.filename}: got {len(raw)} of {info.compress_size} bytes")
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -zlib.MAX_WBITS)
    else:
        raise DownloadError(f"{info.filename}: unsupported compression {info.compress_type}")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ChecksumError(f"{info.filename}: size or CRC32 mismatch")
    return data


def fetch_member(
    transport: Transport, retry: RetryPolicy, url: str, archive_bytes: int, info: zipfile.ZipInfo
) -> bytes:
    end = info.header_offset + HEADER_SLACK + len(info.filename) + info.compress_size
    end = min(end, archive_bytes - 1)
    span = read_range(transport, retry, url, info.header_offset, end)
    return member_from_span(info, span)


def fetch_members(
    transport: Transport,
    retry: RetryPolicy,
    url: str,
    archive_bytes: int,
    infos: Iterable[zipfile.ZipInfo],
    workers: int = 8,
) -> dict[str, bytes]:
    """Members by name, fetched with up to `workers` concurrent range requests."""
    selected = list(infos)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        datas = pool.map(lambda i: fetch_member(transport, retry, url, archive_bytes, i), selected)
        return {info.filename: data for info, data in zip(selected, datas, strict=True)}
