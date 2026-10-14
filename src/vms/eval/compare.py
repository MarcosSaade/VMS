"""Result-set comparison for structured-query evaluation (execution accuracy).

Two results match when they hold the same rows: column names are ignored (a planner may call
a count `n` or `people`), numbers match within a tolerance (an average computed in another
order differs in the last bits), and row order matters only when the question implies one
(`ordered`: top-k, trends, "the latest"). The predicted result may carry extra columns (a list
of tracks returns ids, times and counts where the gold SQL selected ids only): each expected
column must map to a distinct predicted column, and the rows must match on that projection.

What this does not measure: whether extra columns are sensible, how the answer is phrased,
and whether a match is a coincidence (two different queries agreeing on a small result, such
as a count of 0). The scorer (task 3.2) reports those limits next to the metric.

Values are normalized before comparison and before gold results are stored as JSON:
Decimal -> float, aware datetimes -> UTC ISO strings, naive datetimes (camera-local trend
buckets) -> ISO strings, timedelta -> seconds.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

# Cap on column mappings tried; results here have at most ~10 columns, so this is never hit
# by real data, only by adversarial shapes (many identical columns).
MAX_MAPPINGS = 5_000


class ResultSet(BaseModel):
    """Columns and rows as JSON-ready values (see `normalize_value`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: list[str]
    rows: list[list[Any]]


def normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            return value.astimezone(dt.UTC).isoformat()
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    return value


def result_set(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> ResultSet:
    return ResultSet(
        columns=list(columns), rows=[[normalize_value(v) for v in row] for row in rows]
    )


# Decimal places floats are stored with in gold results. Postgres may sum floats in another order
# between runs (parallel workers), changing an average in the last bits; rounding keeps the
# stored files, and so the test split's hash, identical across rebuilds.
STORED_DECIMALS = 9


def canonical(result: ResultSet, *, ordered: bool) -> ResultSet:
    """The form gold results are stored in: floats rounded, and rows sorted unless their order
    is part of the answer (the database returns unordered rows in any order)."""
    rows = [
        [round(v, STORED_DECIMALS) if isinstance(v, float) else v for v in row]
        for row in result.rows
    ]
    if not ordered:
        rows.sort(key=_sort_key)
    return ResultSet(columns=result.columns, rows=rows)


@dataclass(frozen=True)
class Tolerance:
    rel: float = 1e-6
    abs: float = 1e-6


@dataclass(frozen=True)
class Match:
    ok: bool
    reason: str = ""  # why not, for error analysis


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def values_equal(a: Any, b: Any, tol: Tolerance) -> bool:
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=tol.rel, abs_tol=tol.abs)
    return type(a) is type(b) and bool(a == b)


def _sort_key(row: Sequence[Any]) -> tuple[Any, ...]:
    """Total order over mixed cells; numbers are rounded so near-equal values sort together."""
    key: list[tuple[int, Any]] = []
    for v in row:
        if v is None:
            key.append((0, 0))
        elif _is_number(v):
            key.append((1, round(float(v), 6)))
        else:
            key.append((2, str(v)))
    return tuple(key)


def _rows_equal(
    expected: list[list[Any]], actual: list[list[Any]], ordered: bool, tol: Tolerance
) -> bool:
    if not ordered:
        expected, actual = sorted(expected, key=_sort_key), sorted(actual, key=_sort_key)
    return all(
        all(values_equal(e, a, tol) for e, a in zip(er, ar, strict=True))
        for er, ar in zip(expected, actual, strict=True)
    )


def _column(rows: list[list[Any]], j: int) -> list[Any]:
    return [row[j] for row in rows]


def compare(
    expected: ResultSet, actual: ResultSet, *, ordered: bool, tol: Tolerance | None = None
) -> Match:
    tol = tol or Tolerance()
    if len(expected.rows) != len(actual.rows):
        return Match(False, f"{len(actual.rows)} rows, expected {len(expected.rows)}")
    if not expected.rows:
        return Match(True)
    n_expected, n_actual = len(expected.rows[0]), len(actual.rows[0])
    if n_actual < n_expected:
        return Match(False, f"{n_actual} columns, expected at least {n_expected}")

    # Candidate predicted columns per expected column: the same values as a multiset. Order is
    # checked on whole rows below, so a reordered result is reported as such.
    candidates: list[list[int]] = []
    for i in range(n_expected):
        want = [[v] for v in _column(expected.rows, i)]
        found = [
            j
            for j in range(n_actual)
            if _rows_equal(want, [[v] for v in _column(actual.rows, j)], False, tol)
        ]
        if not found:
            name = expected.columns[i] if i < len(expected.columns) else str(i)
            return Match(False, f"no column matches expected column {name!r}")
        candidates.append(found)

    for tried, mapping in enumerate(itertools.product(*candidates)):
        if tried >= MAX_MAPPINGS:
            return Match(False, "too many ambiguous column mappings")
        if len(set(mapping)) != len(mapping):
            continue
        projected = [[row[j] for j in mapping] for row in actual.rows]
        if _rows_equal(expected.rows, projected, ordered, tol):
            return Match(True)
        if ordered and _rows_equal(expected.rows, projected, False, tol):
            return Match(False, "same rows in a different order")
    return Match(False, "rows differ")
