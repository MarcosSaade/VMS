"""query_log stages: per-stage timings, row count and model on every logged query (task 1.7)

The query service logs one row per question (ARCHITECTURE 4.7). Schema v1 had the total
latency only; the stage breakdown (catalog, plan, compile, validate, execute) is what tells a
slow planner from a slow query, so it is stored with each row. `llm` records which model (or
the canned client) produced the plan, so eval and demo rows can be told apart.

query_log stays out of the read-only role's grants: the query service writes it through its
own connection as the schema owner, never through the reader that runs compiled SQL.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-12
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    ALTER TABLE query_log
        ADD COLUMN timings    jsonb,                  -- {"plan": ms, "compile": ms, ...}
        ADD COLUMN row_count  integer CHECK (row_count >= 0),
        ADD COLUMN llm        text                    -- e.g. openai:qwen2.5:7b, canned
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE query_log DROP COLUMN timings, DROP COLUMN row_count, DROP COLUMN llm")
