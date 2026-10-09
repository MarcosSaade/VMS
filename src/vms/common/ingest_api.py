"""The ingest REST contract for frames, shared by loadgen (client) and ingest (server).

    POST /v1/frames    multipart/form-data
      camera_id    text   stable camera identifier, e.g. "cam-001"
      run_id       text   identifies one sender session (a loadgen run, a camera boot)
      seq          int    per (camera_id, run_id), starts at 0, +1 per captured frame
      ts_unix_ms   int    capture time, Unix milliseconds (UTC)
      jpeg         file   the frame, image/jpeg

Status codes and what a client should make of them:
    200/201/202  accepted (a duplicate of an already stored frame is also accepted, so a retry
                 after a lost response is safe)
    429, 503     rejected: the server is shedding load (backpressure); the frame is not stored
    other        failed: client bug (4xx) or server error (5xx)

`seq` restarts at 0 whenever a sender restarts, so `(camera_id, seq)` alone is not unique
across runs; `run_id` disambiguates. Idempotency keys must include it.

`camera_id` and `run_id` are limited to letters, digits, `.`, `_` and `-` (not starting with
`.`): they become parts of object-store keys, so `/`, `..` and control characters must never get
through.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

FRAMES_PATH: Final = "/v1/frames"
JPEG_FIELD: Final = "jpeg"
JPEG_CONTENT_TYPE: Final = "image/jpeg"
REJECT_STATUSES: Final = frozenset({429, 503})
SAFE_ID_PATTERN: Final = r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}$"


class FrameMeta(BaseModel):
    """The non-image form fields of a frame upload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(pattern=SAFE_ID_PATTERN)
    run_id: str = Field(pattern=SAFE_ID_PATTERN)
    seq: int = Field(ge=0)
    ts_unix_ms: int = Field(ge=0)


class FrameAccepted(BaseModel):
    """Body of a 2xx response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str
    duplicate: bool = False
