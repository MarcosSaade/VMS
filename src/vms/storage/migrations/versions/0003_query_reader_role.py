"""query_reader: SELECT-only group role for the query service (task 1.4)

The query service logs in as POSTGRES_READONLY_USER, a member of `vms_query_reader` created by
`python -m vms.storage bootstrap` (make migrate), because a login password does not belong in a
migration. This role holds the privileges: SELECT on exactly the columns in the SQL validator's
allowlist (`configs/query.yaml`, `sql.tables`), granted column by column. So the allowlist is
enforced twice: by the validator before execution and by Postgres during it. Anything not
granted (query_log, frames.status, detections.det_idx, every write, every table added later)
fails with a permission error even if a query slips past the validator. TEMPORARY on the
database is revoked from PUBLIC, so no role without an explicit grant can create temp tables.

The grants are listed here rather than read from the config, so the migration stays fixed
when the config changes; an integration test checks the two agree.

Roles are cluster-wide but grants are per database. The role is created if missing and never
dropped on downgrade: other databases on the server (the dev database, test databases) may
still hold grants to it, and DROP ROLE would fail on them.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-12
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

ROLE = "vms_query_reader"

GRANTS: dict[str, tuple[str, ...]] = {
    "cameras": ("camera_id", "name", "site", "timezone", "created_at"),
    "zones": ("zone_id", "camera_id", "name", "polygon"),
    "frames": (
        "frame_id",
        "camera_id",
        "ts",
        "object_key",
        "width",
        "height",
        "brightness",
        "blur_var",
        "model_version",
    ),
    "detections": (
        "ts",
        "camera_id",
        "frame_id",
        "track_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "model_version",
    ),
    "tracks": ("track_id", "camera_id", "class_name", "first_seen", "last_seen", "n_frames"),
    "events": ("event_id", "ts", "camera_id", "track_id", "zone_id", "event_type"),
}


def upgrade() -> None:
    # duplicate_object: another database's migration created it between the check and CREATE.
    op.execute(f"""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{ROLE}') THEN
            CREATE ROLE {ROLE} NOLOGIN;
        END IF;
    EXCEPTION WHEN duplicate_object THEN NULL;
    END
    $$
    """)
    for table, columns in GRANTS.items():
        op.execute(f"GRANT SELECT ({', '.join(columns)}) ON {table} TO {ROLE}")
    _temp_for_public("REVOKE", "FROM")


def downgrade() -> None:
    _temp_for_public("GRANT", "TO")
    for table in GRANTS:
        op.execute(f"REVOKE ALL ON {table} FROM {ROLE}")


def _temp_for_public(verb: str, preposition: str) -> None:
    # Postgres grants TEMPORARY on every database to PUBLIC; without it a reader cannot create
    # temp tables even after switching its session out of read-only mode. Services that write
    # connect as the schema owner and are unaffected.
    op.execute(f"""
    DO $$
    BEGIN
        EXECUTE format('{verb} TEMPORARY ON DATABASE %I {preposition} PUBLIC', current_database());
    END
    $$
    """)
