"""SQL validator on sqlglot ASTs: the last gate before any SQL reaches the database.

It guards both compiled SQL from the plan engine (defense in depth: the compiler should never
emit anything this rejects) and model-written SQL from the free-form engine (the primary
guard). It is not the only control: SQL also runs on a read-only role with a statement timeout
(ARCHITECTURE 11).

Rules, checked on the parsed tree rather than on text:
- exactly one statement, and it is a SELECT or a set operation (UNION, INTERSECT, EXCEPT);
- no DML or DDL anywhere in the tree (catches data-modifying CTEs such as
  `WITH d AS (DELETE ... RETURNING *) SELECT ...`), no SELECT INTO, no FOR UPDATE/SHARE,
  no recursive CTEs;
- only allowlisted tables, never schema-qualified (blocks pg_catalog, information_schema);
  FROM/JOIN sources are named tables or subqueries only (no table functions, VALUES, LATERAL);
- every column resolves to an allowlisted column (sqlglot `qualify`; blocks system columns such
  as ctid and typos);
- no denied function, matched by name or prefix (pg_sleep, pg_read_file, dblink, ...);
- at most `max_joins` joins;
- the outermost query gets `LIMIT default_limit` if it has none and is capped at `max_limit`.
  LIMIT must be an integer literal, so compilers inline the validated limit.

The returned SQL is regenerated from the checked tree: comments are dropped, `*` is expanded to
allowlisted columns only, and the limit is applied. Execute that, never the input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, TokenError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from vms.query.settings import SqlValidatorSettings


class Rejection(StrEnum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    PARSE_ERROR = "parse_error"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    FORBIDDEN_STATEMENT = "forbidden_statement"
    SELECT_INTO = "select_into"
    LOCKING_CLAUSE = "locking_clause"
    RECURSIVE_CTE = "recursive_cte"
    SCHEMA_QUALIFIED = "schema_qualified"
    FORBIDDEN_TABLE = "forbidden_table"
    FORBIDDEN_FUNCTION = "forbidden_function"
    UNKNOWN_COLUMN = "unknown_column"
    TOO_MANY_JOINS = "too_many_joins"
    BAD_LIMIT = "bad_limit"


class SqlValidationError(ValueError):
    def __init__(self, code: Rejection, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ValidatedSql:
    sql: str  # regenerated, safe to execute
    tables: frozenset[str]  # allowlisted tables read
    limit: int  # effective outer LIMIT


# Node types that must not appear anywhere in a read-only query. exp.DML covers INSERT, UPDATE,
# DELETE and MERGE; exp.DDL covers CREATE, DROP and friends.
_FORBIDDEN_NODES: tuple[type[exp.Expr], ...] = (
    exp.DML,
    exp.DDL,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Use,
)
_FUNC_NAME = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_.]*)")


class SqlValidator:
    def __init__(self, settings: SqlValidatorSettings) -> None:
        self._s = settings
        self._schema: dict[str, object] = {t: dict(cols) for t, cols in settings.tables.items()}
        self._denied = frozenset(settings.denied_functions)
        self._denied_prefixes = tuple(settings.denied_function_prefixes)

    def validate(self, sql: str) -> ValidatedSql:
        root = self._parse_single(sql)
        if not isinstance(root, exp.Select | exp.SetOperation):
            raise SqlValidationError(
                Rejection.NOT_SELECT, f"only SELECT is allowed, got {root.key.upper()}"
            )
        self._check_nodes(root)
        tables = self._check_tables(root)
        self._check_functions(root)
        joins = sum(1 for _ in root.find_all(exp.Join))
        if joins > self._s.max_joins:
            raise SqlValidationError(
                Rejection.TOO_MANY_JOINS, f"{joins} joins, at most {self._s.max_joins} allowed"
            )
        root = self._qualify(root)
        limit = self._apply_limit(root)
        return ValidatedSql(
            sql=root.sql(dialect=self._s.dialect, comments=False),
            tables=frozenset(tables),
            limit=limit,
        )

    def _parse_single(self, sql: str) -> exp.Expr:
        if not sql.strip():
            raise SqlValidationError(Rejection.EMPTY, "empty statement")
        if len(sql) > self._s.max_sql_chars:
            raise SqlValidationError(
                Rejection.TOO_LONG, f"{len(sql)} chars, at most {self._s.max_sql_chars} allowed"
            )
        try:
            statements = [s for s in sqlglot.parse(sql, read=self._s.dialect) if s is not None]
        except (ParseError, TokenError) as exc:
            raise SqlValidationError(Rejection.PARSE_ERROR, str(exc).splitlines()[0]) from exc
        if not statements:
            raise SqlValidationError(Rejection.EMPTY, "no statement")
        if len(statements) > 1:
            raise SqlValidationError(
                Rejection.MULTIPLE_STATEMENTS, f"{len(statements)} statements, exactly 1 allowed"
            )
        # Fold unquoted identifiers the way the database does (lowercase in Postgres), so the
        # allowlist sees the names the server will resolve.
        return normalize_identifiers(statements[0], dialect=self._s.dialect)

    def _check_nodes(self, root: exp.Expr) -> None:
        for node in root.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise SqlValidationError(
                    Rejection.FORBIDDEN_STATEMENT, f"{node.key.upper()} is not allowed"
                )
            if isinstance(node, exp.Into):
                raise SqlValidationError(Rejection.SELECT_INTO, "SELECT INTO is not allowed")
            if isinstance(node, exp.Lock):
                raise SqlValidationError(
                    Rejection.LOCKING_CLAUSE, "FOR UPDATE/SHARE is not allowed"
                )
            if isinstance(node, exp.With) and node.args.get("recursive"):
                raise SqlValidationError(Rejection.RECURSIVE_CTE, "WITH RECURSIVE is not allowed")

    def _check_tables(self, root: exp.Expr) -> set[str]:
        # Every FROM/JOIN source must be a named table or a subquery. Table-valued functions
        # (generate_series, unnest, pg_ls_dir), VALUES lists and LATERAL are not needed on this
        # schema, and functions in FROM are a common route to server internals.
        for clause in root.find_all(exp.From, exp.Join):
            source = clause.this
            named = isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)
            if not (named or isinstance(source, exp.Subquery)):
                raise SqlValidationError(
                    Rejection.FORBIDDEN_TABLE,
                    f"FROM source {source.sql(dialect=self._s.dialect)!r} is not allowed",
                )

        # Resolve names per scope: a CTE is visible only inside the query that defines it, so a
        # global list of CTE names would let `(WITH pg_user AS ...) ... JOIN pg_user` reach the
        # real table. In each scope, a source that resolves to a Table is a real table.
        try:
            scopes = traverse_scope(root)
        except OptimizeError as exc:
            raise SqlValidationError(Rejection.PARSE_ERROR, str(exc).splitlines()[0]) from exc
        real: dict[int, exp.Table] = {}
        for scope in scopes:
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    real[id(source)] = source
        cte_names = {cte.alias_or_name for cte in root.find_all(exp.CTE)}

        tables: set[str] = set()
        for table in root.find_all(exp.Table):
            if table.args.get("db") or table.args.get("catalog"):
                raise SqlValidationError(
                    Rejection.SCHEMA_QUALIFIED,
                    f"schema-qualified table {table.sql(dialect=self._s.dialect)!r} is not allowed",
                )
            if id(table) not in real and table.name in cte_names:
                continue  # a reference to a CTE in scope
            if table.name not in self._schema:
                raise SqlValidationError(
                    Rejection.FORBIDDEN_TABLE, f"table {table.name!r} is not queryable"
                )
            tables.add(table.name)
        return tables

    def _check_functions(self, root: exp.Expr) -> None:
        for func in root.find_all(exp.Func):
            names = self._function_names(func)
            for name in names:
                if name in self._denied or name.startswith(self._denied_prefixes):
                    raise SqlValidationError(
                        Rejection.FORBIDDEN_FUNCTION, f"function {name!r} is not allowed"
                    )

    def _function_names(self, func: exp.Func) -> set[str]:
        """Every name a function may be known by: sqlglot normalizes many functions to its own
        canonical names (generate_series -> EXPLODING_GENERATE_SERIES), so match the name as
        written, the canonical name and the name rendered back in the dialect."""
        names = {func.sql_name().lower()}
        if isinstance(func, exp.Anonymous):
            names.add(func.name.lower())
        match = _FUNC_NAME.match(func.sql(dialect=self._s.dialect))
        if match:
            names.add(match.group(1).lower())
        return names

    def _qualify(self, root: exp.Expr) -> exp.Expr:
        try:
            return qualify(
                root.copy(),
                schema=self._schema,
                dialect=self._s.dialect,
                validate_qualify_columns=True,
                identify=False,
            )
        except OptimizeError as exc:
            raise SqlValidationError(Rejection.UNKNOWN_COLUMN, str(exc).splitlines()[0]) from exc

    def _apply_limit(self, root: exp.Expr) -> int:
        current = root.args.get("limit")
        if current is None:
            limit = self._s.default_limit
        else:
            if not isinstance(current, exp.Limit):
                raise SqlValidationError(Rejection.BAD_LIMIT, "use LIMIT <n>, not FETCH FIRST")
            value = current.expression
            if not (isinstance(value, exp.Literal) and value.is_int):
                raise SqlValidationError(Rejection.BAD_LIMIT, "LIMIT must be an integer literal")
            limit = min(int(value.this), self._s.max_limit)
        root.set("limit", exp.Limit(expression=exp.Literal.number(limit)))
        return limit
