import pytest

from vms.query.settings import SqlValidatorSettings
from vms.query.sql_validator import Rejection, SqlValidationError, SqlValidator


def rejects(validator: SqlValidator, sql: str) -> Rejection:
    with pytest.raises(SqlValidationError) as excinfo:
        validator.validate(sql)
    return excinfo.value.code


# --- Accepted queries ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM detections",
        "SELECT class_name, count(*) AS n FROM detections GROUP BY class_name ORDER BY n DESC",
        "SELECT d.class_name FROM detections d JOIN cameras c ON c.camera_id = d.camera_id "
        "WHERE c.name = 'entrance'",
        "SELECT t.track_id FROM tracks t WHERE t.track_id IN "
        "(SELECT e.track_id FROM events e WHERE e.event_type = 'zone_enter')",
        "WITH per_cam AS (SELECT camera_id, count(*) AS n FROM tracks GROUP BY camera_id) "
        "SELECT c.name, p.n FROM per_cam p JOIN cameras c USING (camera_id)",
        "SELECT camera_id FROM tracks UNION SELECT camera_id FROM events",
        "SELECT date_trunc('hour', ts) AS h, count(*) FROM events GROUP BY 1 ORDER BY 1",
        "SELECT avg(EXTRACT(EPOCH FROM last_seen - first_seen)) FROM tracks",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY confidence) FROM detections",
        "SELECT camera_id, count(*) OVER (PARTITION BY camera_id) FROM tracks",
        "SELECT * FROM detections WHERE ts >= now() - INTERVAL '2 hours'",
        "SELECT count(*) FROM detections d JOIN cameras c ON c.camera_id = d.camera_id "
        "WHERE d.ts >= date_trunc('day', now() AT TIME ZONE c.timezone) AT TIME ZONE c.timezone",
        "SELECT frame_id FROM frames WHERE camera_id = %(camera)s AND ts >= %(start)s",
        "SELECT 1 FROM cameras;",
        "SELECT name FROM cameras WHERE name = 'x; DROP TABLE cameras'",
        "SELECT $$it's fine$$ AS s FROM cameras",
        "SELECT x FROM (SELECT class_name AS x FROM detections) AS s",
        'SELECT "name" FROM "cameras"',
        "sElEcT NaMe FrOm CaMeRaS",
        "SELECT name FROM cameras WHERE name = E'\\x27; DROP TABLE cameras; --'",
    ],
)
def test_accepts_read_only_queries(validator: SqlValidator, sql: str) -> None:
    result = validator.validate(sql)
    assert result.sql
    assert result.limit == 100


def test_reports_tables_read(validator: SqlValidator) -> None:
    result = validator.validate(
        "WITH x AS (SELECT track_id FROM events) "
        "SELECT t.class_name FROM tracks t JOIN x ON x.track_id = t.track_id"
    )
    assert result.tables == {"events", "tracks"}


def test_string_literals_are_data_not_sql(validator: SqlValidator) -> None:
    result = validator.validate("SELECT name FROM cameras WHERE name = 'a''; DELETE FROM x; --'")
    assert result.tables == {"cameras"}
    assert "'a''; DELETE FROM x; --'" in result.sql


def test_placeholders_survive_regeneration(validator: SqlValidator) -> None:
    result = validator.validate("SELECT frame_id FROM frames WHERE camera_id = %(camera)s")
    assert "%(camera)s" in result.sql


# --- Statement shape -------------------------------------------------------------------------


@pytest.mark.parametrize("sql", ["", "   ", "\n\t", ";", ";;"])
def test_rejects_empty(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.EMPTY


def test_rejects_too_long() -> None:
    small = SqlValidator(SqlValidatorSettings(tables={"t": {"a": "int"}}, max_sql_chars=30))
    assert rejects(small, "SELECT a FROM t WHERE a = 1 OR a = 2") is Rejection.TOO_LONG


@pytest.mark.parametrize("sql", ["SELEC 1", "SELECT FROM WHERE", "SELECT (1", "SELECT 'open"])
def test_rejects_unparsable(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.PARSE_ERROR


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM cameras; DROP TABLE cameras",
        "SELECT 1 FROM cameras; SELECT 2 FROM cameras",
        "SELECT name FROM cameras WHERE name = '';DELETE FROM cameras;--'",
        "SELECT 1 FROM cameras;\nUPDATE cameras SET name = 'x'",
    ],
)
def test_rejects_multiple_statements(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.MULTIPLE_STATEMENTS


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE detections",
        "TRUNCATE detections",
        "DELETE FROM detections",
        "UPDATE cameras SET name = 'x'",
        "INSERT INTO cameras (name) VALUES ('x')",
        "CREATE TABLE x (a int)",
        "ALTER TABLE cameras ADD COLUMN pwned text",
        "CREATE ROLE evil SUPERUSER",
        "GRANT ALL ON detections TO public",
        "COPY cameras TO '/tmp/out.csv'",
        "SET statement_timeout = 0",
        "SET ROLE postgres",
        "BEGIN",
        "COMMIT",
        "VACUUM detections",
        "EXPLAIN ANALYZE SELECT 1",
        "DO $$ BEGIN PERFORM pg_sleep(10); END $$",
        "CALL some_procedure()",
        "SHOW ALL",
        "TABLE cameras",
        "VALUES (1), (2)",
        "MERGE INTO cameras c USING tracks t ON c.camera_id = t.camera_id WHEN MATCHED THEN DELETE",
    ],
)
def test_rejects_non_select_statements(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) in {
        Rejection.NOT_SELECT,
        Rejection.FORBIDDEN_STATEMENT,
        Rejection.PARSE_ERROR,
    }


@pytest.mark.parametrize(
    "sql",
    [
        "WITH d AS (DELETE FROM detections RETURNING *) SELECT count(*) FROM d",
        "WITH u AS (UPDATE cameras SET name = 'x' RETURNING camera_id) SELECT * FROM u",
        "WITH i AS (INSERT INTO cameras (name) VALUES ('x') RETURNING camera_id) SELECT * FROM i",
    ],
)
def test_rejects_data_modifying_ctes(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.FORBIDDEN_STATEMENT


def test_rejects_select_into(validator: SqlValidator) -> None:
    assert rejects(validator, "SELECT * INTO stolen FROM cameras") is Rejection.SELECT_INTO


@pytest.mark.parametrize("lock", ["FOR UPDATE", "FOR SHARE", "FOR NO KEY UPDATE"])
def test_rejects_locking_clauses(validator: SqlValidator, lock: str) -> None:
    assert rejects(validator, f"SELECT name FROM cameras {lock}") is Rejection.LOCKING_CLAUSE


def test_rejects_recursive_cte(validator: SqlValidator) -> None:
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r"
    assert rejects(validator, sql) is Rejection.RECURSIVE_CTE


# --- Tables and columns ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM query_log",
        "SELECT * FROM embeddings",
        "SELECT * FROM pg_user",
        "SELECT * FROM pg_shadow",
        "SELECT usename FROM pg_stat_activity",
        "SELECT name FROM cameras UNION SELECT usename FROM pg_user",
        "SELECT name FROM cameras WHERE camera_id IN (SELECT rolname FROM pg_roles)",
        "SELECT name FROM cameras c JOIN secrets s ON true",
        'SELECT * FROM "pg_user"',
        "SELECT * FROM PG_USER",
    ],
)
def test_rejects_unlisted_tables(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.FORBIDDEN_TABLE


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT table_name FROM information_schema.tables",
        "SELECT * FROM public.cameras",
        "SELECT * FROM otherdb.public.cameras",
    ],
)
def test_rejects_schema_qualified_tables(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.SCHEMA_QUALIFIED


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM generate_series(1, 1000000000)",
        "SELECT * FROM pg_ls_dir('.')",
        "SELECT * FROM unnest(ARRAY[1, 2])",
        "SELECT * FROM (VALUES (1), (2)) AS v(a)",
        "SELECT c.name FROM cameras c CROSS JOIN LATERAL pg_ls_dir('.') AS f",
        "SELECT c.name FROM cameras c JOIN generate_series(1, 9) AS g ON true",
    ],
)
def test_rejects_table_functions(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) in {Rejection.FORBIDDEN_TABLE, Rejection.FORBIDDEN_FUNCTION}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT password FROM cameras",
        "SELECT ctid FROM detections",
        "SELECT xmin FROM cameras",
        "SELECT name FROM cameras WHERE secret = 1",
        "SELECT c.vector FROM cameras c",
        "SELECT confidence FROM tracks",
    ],
)
def test_rejects_unknown_columns(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.UNKNOWN_COLUMN


def test_star_expands_to_allowlisted_columns_only() -> None:
    narrow = SqlValidator(SqlValidatorSettings(tables={"cameras": {"camera_id": "text"}}))
    result = narrow.validate("SELECT * FROM cameras")
    assert "*" not in result.sql
    assert "camera_id" in result.sql


def test_cte_names_do_not_unlock_real_tables(validator: SqlValidator) -> None:
    # A CTE named like a forbidden table is fine; reading the real one inside it is not.
    ok = "WITH pg_user AS (SELECT name FROM cameras) SELECT name FROM pg_user"
    assert validator.validate(ok).tables == {"cameras"}
    bad = "WITH x AS (SELECT usename FROM pg_user) SELECT usename FROM x"
    assert rejects(validator, bad) is Rejection.FORBIDDEN_TABLE


@pytest.mark.parametrize(
    "sql",
    [
        # A CTE is visible only in its own query: the outer pg_user is the real table.
        "SELECT s.name FROM (WITH pg_user AS (SELECT name FROM cameras) SELECT name FROM pg_user)"
        " AS s CROSS JOIN pg_user",
        "SELECT name FROM cameras WHERE EXISTS (WITH pg_shadow AS (SELECT 1 AS a FROM cameras)"
        " SELECT a FROM pg_shadow) AND EXISTS (SELECT 1 FROM pg_shadow)",
    ],
)
def test_cte_shadowing_is_scoped(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.FORBIDDEN_TABLE


# --- Functions -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(30)",
        "SELECT name FROM cameras WHERE pg_sleep(30) IS NULL",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_ls_dir('.')",
        "SELECT pg_terminate_backend(1)",
        "SELECT PG_SLEEP(1)",
        "SELECT pg_catalog.pg_sleep(1)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT dblink_exec('host=evil', 'DROP TABLE cameras')",
        "SELECT current_setting('data_directory')",
        "SELECT set_config('statement_timeout', '0', false)",
        "SELECT version()",
        "SELECT nextval('cameras_seq')",
        "SELECT query_to_xml('SELECT * FROM pg_shadow', true, true, '')",
        "SELECT generate_series(1, 1000000000)",
        "SELECT repeat('x', 1000000000)",
        "SELECT txid_current()",
        "SELECT count(*) FROM detections WHERE class_name = (SELECT pg_sleep(5)::text)",
    ],
)
def test_rejects_denied_functions(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.FORBIDDEN_FUNCTION


# --- Comments --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM cameras -- ; DROP TABLE cameras",
        "SELECT name /* ; DROP TABLE cameras; */ FROM cameras",
        "/* leading */ SELECT name FROM cameras",
        "SELECT name FROM cameras WHERE name = 'a' --'",
    ],
)
def test_comments_are_stripped_and_cannot_smuggle_statements(
    validator: SqlValidator, sql: str
) -> None:
    result = validator.validate(sql)
    assert "--" not in result.sql
    assert "/*" not in result.sql
    assert "DROP" not in result.sql.upper()


def test_comment_hiding_a_second_statement_still_counts(validator: SqlValidator) -> None:
    sql = "SELECT name FROM cameras /* harmless */; DROP TABLE cameras"
    assert rejects(validator, sql) is Rejection.MULTIPLE_STATEMENTS


# --- Joins and limits ------------------------------------------------------------------------


def test_rejects_too_many_joins(validator: SqlValidator) -> None:
    joins = " ".join(f"CROSS JOIN tracks t{i}" for i in range(5))
    assert rejects(validator, f"SELECT 1 FROM tracks t {joins}") is Rejection.TOO_MANY_JOINS


def test_injects_default_limit(validator: SqlValidator) -> None:
    result = validator.validate("SELECT name FROM cameras")
    assert result.limit == 100
    assert result.sql.endswith("LIMIT 100")


def test_keeps_smaller_limit_and_caps_larger(validator: SqlValidator) -> None:
    assert validator.validate("SELECT name FROM cameras LIMIT 5").limit == 5
    capped = validator.validate("SELECT name FROM cameras LIMIT 999999999")
    assert capped.limit == 1000
    assert capped.sql.endswith("LIMIT 1000")


def test_limit_applies_to_outer_set_operation(validator: SqlValidator) -> None:
    result = validator.validate("SELECT camera_id FROM tracks UNION SELECT camera_id FROM events")
    assert result.sql.endswith("LIMIT 100")
    assert result.sql.count("LIMIT") == 1


def test_inner_limits_do_not_count_as_outer(validator: SqlValidator) -> None:
    result = validator.validate(
        "SELECT name FROM cameras WHERE camera_id IN (SELECT camera_id FROM tracks LIMIT 5000)"
    )
    assert result.sql.endswith("LIMIT 100")


def test_offset_is_kept(validator: SqlValidator) -> None:
    result = validator.validate("SELECT name FROM cameras ORDER BY name LIMIT 10 OFFSET 20")
    assert "OFFSET 20" in result.sql
    assert result.limit == 10


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM cameras LIMIT %(n)s",
        "SELECT name FROM cameras LIMIT (SELECT count(*) FROM tracks)",
        "SELECT name FROM cameras LIMIT 1.5",
        "SELECT name FROM cameras LIMIT -1",
        "SELECT name FROM cameras FETCH FIRST 5 ROWS ONLY",
    ],
)
def test_rejects_non_literal_limits(validator: SqlValidator, sql: str) -> None:
    assert rejects(validator, sql) is Rejection.BAD_LIMIT


def test_error_carries_code_and_detail(validator: SqlValidator) -> None:
    with pytest.raises(SqlValidationError, match="forbidden_table") as excinfo:
        validator.validate("SELECT * FROM pg_user")
    assert "pg_user" in excinfo.value.detail


def test_settings_reject_default_above_max() -> None:
    with pytest.raises(ValueError, match="default_limit"):
        SqlValidatorSettings(default_limit=10, max_limit=5)
