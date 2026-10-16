"""drift: a noise estimate on each frame and per-camera drift references (task 9.1)

The drift monitor (vms.drift) compares each camera's recent window with a reference window
captured during a calibration period. `frames.noise` is the worker's new input statistic
(vms.worker.stats; NULL on frames processed before this migration). `drift_references` holds
one calibration period per camera, as a capture-time range over the camera's own frames and
detections: the monitor recomputes the reference from those rows after a restart, and
`python -m vms.drift recalibrate` replaces a range (ARCHITECTURE 4.9). The range must stay
within retained data; there is no retention yet.

Neither is granted to the query reader: drift state is operations data, not a queryable fact.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-16
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE frames ADD COLUMN noise double precision")
    op.execute("""
    CREATE TABLE drift_references (
        camera_id   text PRIMARY KEY,
        start_ts    timestamptz NOT NULL,             -- capture time, inclusive
        end_ts      timestamptz NOT NULL,             -- capture time, exclusive
        created_at  timestamptz NOT NULL DEFAULT now(),
        reason      text NOT NULL CHECK (reason IN ('auto', 'recalibrate')),
        CHECK (end_ts > start_ts)
    )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE drift_references")
    op.execute("ALTER TABLE frames DROP COLUMN noise")
