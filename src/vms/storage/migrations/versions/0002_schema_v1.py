"""schema v1: cameras, zones, detections, tracks, events, query_log (task 1.4)

ARCHITECTURE 4.6 without `embeddings` (CLIP is deferred). Conventions (D19):

- Pipeline-written tables (frames, detections, tracks, events) have no foreign keys: they are
  the D1 detection-store candidates, and column stores have none; ingest also accepts frames
  from cameras not yet in the catalog (load tests use any camera id). Only zones -> cameras,
  both written from config, is enforced.
- detections and events are append-only; tracks is updated in place as a track grows.
- Indexes are the minimum for the baseline: `(camera_id, ts)` on every time-series table,
  because every structured query filters by camera and time.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE cameras (
        camera_id   text PRIMARY KEY,                -- the id cameras send frames with
        name        text NOT NULL UNIQUE,            -- what questions and plans call it
        site        text NOT NULL,
        timezone    text NOT NULL,                   -- IANA name, for "this morning" questions
        created_at  timestamptz NOT NULL DEFAULT now()
    )
    """)
    op.execute("""
    CREATE TABLE zones (
        zone_id    text PRIMARY KEY,                 -- '<camera_id>:<name>'
        camera_id  text NOT NULL REFERENCES cameras (camera_id) ON DELETE CASCADE,
        name       text NOT NULL,
        -- [[x, y], ...] in frame coordinates normalized to 0..1, so zones survive resizing.
        polygon    jsonb NOT NULL CHECK (jsonb_typeof(polygon) = 'array'
                                         AND jsonb_array_length(polygon) >= 3),
        CONSTRAINT zones_camera_name UNIQUE (camera_id, name)
    )
    """)
    op.execute("""
    CREATE TABLE detections (
        frame_id       text NOT NULL,
        det_idx        smallint NOT NULL CHECK (det_idx >= 0),  -- position in the frame's output
        ts             timestamptz NOT NULL,          -- the frame's capture time
        camera_id      text NOT NULL,
        track_id       text,                          -- set by the tracker (task 1.6)
        class_name     text NOT NULL,
        confidence     double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
        -- Box corners in pixels of the frame.
        x1             double precision NOT NULL,
        y1             double precision NOT NULL,
        x2             double precision NOT NULL,
        y2             double precision NOT NULL,
        model_version  text NOT NULL,
        -- A redelivered frame rewrites the same keys, so its detections are not doubled.
        PRIMARY KEY (frame_id, det_idx),
        CHECK (x1 <= x2 AND y1 <= y2)
    )
    """)
    op.execute("CREATE INDEX detections_camera_ts ON detections (camera_id, ts)")
    op.execute("""
    CREATE TABLE tracks (
        track_id    text PRIMARY KEY,
        camera_id   text NOT NULL,
        class_name  text NOT NULL,
        first_seen  timestamptz NOT NULL,
        last_seen   timestamptz NOT NULL,
        n_frames    integer NOT NULL CHECK (n_frames >= 1),
        CHECK (first_seen <= last_seen)
    )
    """)
    op.execute("CREATE INDEX tracks_camera_last_seen ON tracks (camera_id, last_seen)")
    op.execute("""
    CREATE TABLE events (
        event_id    text PRIMARY KEY,                 -- ULID
        ts          timestamptz NOT NULL,
        camera_id   text NOT NULL,
        track_id    text NOT NULL,
        zone_id     text NOT NULL,
        event_type  text NOT NULL CHECK (event_type IN ('zone_enter', 'zone_exit', 'loiter'))
    )
    """)
    op.execute("CREATE INDEX events_camera_ts ON events (camera_id, ts)")
    op.execute("""
    CREATE TABLE query_log (
        query_id       text PRIMARY KEY,              -- ULID
        ts             timestamptz NOT NULL DEFAULT now(),
        mode           text NOT NULL,                 -- structured (semantic is deferred)
        engine         text NOT NULL,                 -- plan (free-form SQL is deferred)
        question       text NOT NULL,
        query_plan     jsonb,                         -- null if planning failed
        compiled_sql   text,
        latency_ms     double precision NOT NULL CHECK (latency_ms >= 0),
        tokens         integer CHECK (tokens >= 0),
        success        boolean NOT NULL,
        error_type     text,                          -- null on success
        feedback       text,
        CHECK (success = (error_type IS NULL))
    )
    """)
    op.execute("CREATE INDEX query_log_ts ON query_log (ts)")
    # Queries list frames by camera and time (few-shot bank); the worker's claim index is 1.5's.
    op.execute("CREATE INDEX frames_camera_ts ON frames (camera_id, ts)")


def downgrade() -> None:
    op.execute("DROP INDEX frames_camera_ts")
    op.execute("DROP TABLE query_log, events, tracks, detections, zones, cameras")
