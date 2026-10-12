"""worker claims: claim columns on frames, camera leases (task 1.5)

The frames table is the work queue (D3 default). A worker processes a camera only while it
holds that camera's lease in `camera_leases`, because the tracker's state for a camera lives
in one worker's memory (task 1.6): frames of one camera must keep going to the same worker, in
order, not merely be claimed without overlap. Leases expire unless renewed, so a crashed
worker's cameras move to another worker after `lease_ttl_s` (D20).

Within its cameras, a worker claims frames with `FOR UPDATE SKIP LOCKED` and stamps
`claimed_by`; completion is fenced on that stamp, so a worker that lost its lease cannot write.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-12
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    ALTER TABLE frames
        ADD COLUMN contrast     double precision,     -- input statistic (std of luma)
        ADD COLUMN claimed_by   text,                 -- worker id while status = processing
        ADD COLUMN claimed_at   timestamptz,
        ADD COLUMN attempts     smallint NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        ADD COLUMN processed_at timestamptz,          -- when status became done or failed
        ADD COLUMN error        text                  -- last failure, for failed frames
    """)
    # The claim query reads pending frames of a few cameras in capture order.
    op.execute(
        "CREATE INDEX frames_pending ON frames (camera_id, ts, seq) WHERE status = 'pending'"
    )
    op.execute("""
    CREATE TABLE camera_leases (
        camera_id    text PRIMARY KEY,
        worker_id    text NOT NULL,
        acquired_at  timestamptz NOT NULL DEFAULT now(),
        expires_at   timestamptz NOT NULL
    )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE camera_leases")
    op.execute("DROP INDEX frames_pending")
    op.execute("""
    ALTER TABLE frames
        DROP COLUMN contrast, DROP COLUMN claimed_by, DROP COLUMN claimed_at,
        DROP COLUMN attempts, DROP COLUMN processed_at, DROP COLUMN error
    """)
