"""frames: one row per ingested frame; also the work queue (task 1.3)

The rest of the logical schema (ARCHITECTURE 4.6) arrives in task 1.4.

Revision ID: 0001
Revises:
Create Date: 2026-10-09
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE frames (
        frame_id      text PRIMARY KEY,               -- ULID assigned by ingest
        camera_id     text NOT NULL,
        run_id        text NOT NULL,                  -- sender session (D17)
        seq           bigint NOT NULL CHECK (seq >= 0),
        ts            timestamptz NOT NULL,           -- capture time
        received_at   timestamptz NOT NULL DEFAULT now(),
        object_key    text NOT NULL,
        size_bytes    integer NOT NULL CHECK (size_bytes > 0),
        status        text NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'processing', 'done', 'failed')),
        -- Filled by the worker (task 1.5).
        width         integer,
        height        integer,
        brightness    double precision,
        blur_var      double precision,
        model_version text,
        -- Idempotency key: a retried upload maps to the same row.
        CONSTRAINT frames_idempotency_key UNIQUE (camera_id, run_id, seq)
    )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE frames")
