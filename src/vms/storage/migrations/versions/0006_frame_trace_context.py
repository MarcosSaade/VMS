"""frame trace context: the ingest span's W3C traceparent on each frame (task 2.1)

The frames table is the queue between ingest and the worker (D3 default), so there is no
message header to carry a trace across it. Ingest stores its span's `traceparent`
(`00-<trace id>-<span id>-<flags>`, 55 characters) on the row, and the worker continues that
trace when it claims the frame: one trace then runs loadgen -> ingest -> queue -> worker -> DB
write (D25). The flags carry the sampling decision, so the worker keeps or drops a frame's spans
exactly as loadgen and ingest did. NULL when the sender sent no trace context and ingest did
not start one.

Not granted to the query reader: it is plumbing, not data (D19, configs/query.yaml).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-13
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE frames ADD COLUMN traceparent text CHECK (length(traceparent) <= 64)")


def downgrade() -> None:
    op.execute("ALTER TABLE frames DROP COLUMN traceparent")
