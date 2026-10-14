"""Result comparison: order only when asked, numeric tolerance, column names ignored, extra
predicted columns allowed, and the normalization gold results are stored with."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest

from vms.eval.compare import (
    ResultSet,
    Tolerance,
    canonical,
    compare,
    normalize_value,
    result_set,
)


def rs(columns: list[str], rows: list[list[Any]]) -> ResultSet:
    return ResultSet(columns=columns, rows=rows)


def test_row_order_is_ignored_unless_ordered() -> None:
    expected = rs(["camera", "n"], [["entrance", 3], ["lobby", 5]])
    actual = rs(["camera", "n"], [["lobby", 5], ["entrance", 3]])
    assert compare(expected, actual, ordered=False).ok
    match = compare(expected, actual, ordered=True)
    assert not match.ok
    assert match.reason == "same rows in a different order"


def test_column_names_are_ignored_and_columns_may_be_permuted() -> None:
    expected = rs(["camera", "n"], [["entrance", 3], ["lobby", 5]])
    actual = rs(["people", "cam"], [[5, "lobby"], [3, "entrance"]])
    assert compare(expected, actual, ordered=True).reason == "same rows in a different order"
    assert compare(expected, actual, ordered=False).ok


def test_extra_predicted_columns_are_allowed() -> None:
    expected = rs(["track_id"], [["t1"], ["t2"]])
    actual = rs(["track_id", "camera", "n_frames"], [["t2", "lobby", 4], ["t1", "lobby", 9]])
    assert compare(expected, actual, ordered=False).ok


def test_missing_columns_fail() -> None:
    expected = rs(["camera", "n"], [["lobby", 5]])
    match = compare(expected, rs(["n"], [[5]]), ordered=False)
    assert not match.ok
    assert match.reason == "1 columns, expected at least 2"


def test_rows_must_align_across_columns() -> None:
    # Each column matches as a multiset, but the pairs differ.
    expected = rs(["camera", "n"], [["entrance", 3], ["lobby", 5]])
    actual = rs(["camera", "n"], [["entrance", 5], ["lobby", 3]])
    match = compare(expected, actual, ordered=False)
    assert not match.ok
    assert match.reason == "rows differ"


def test_numbers_match_within_tolerance_and_across_int_and_float() -> None:
    expected = rs(["value"], [[0.8183040799333]])
    assert compare(expected, rs(["v"], [[0.81830408]]), ordered=False).ok
    assert not compare(expected, rs(["v"], [[0.8184]]), ordered=False).ok
    assert compare(rs(["n"], [[3]]), rs(["n"], [[3.0]]), ordered=False).ok
    loose = Tolerance(rel=0.01, abs=0.0)
    assert compare(expected, rs(["v"], [[0.82]]), ordered=False, tol=loose).ok


def test_types_other_than_numbers_must_match_exactly() -> None:
    assert not compare(rs(["n"], [[1]]), rs(["n"], [["1"]]), ordered=False).ok
    assert not compare(rs(["b"], [[True]]), rs(["b"], [[1]]), ordered=False).ok
    assert compare(rs(["v"], [[None]]), rs(["v"], [[None]]), ordered=False).ok
    assert not compare(rs(["v"], [[None]]), rs(["v"], [[0]]), ordered=False).ok


def test_row_counts_and_empty_results() -> None:
    match = compare(rs(["n"], [[1], [2]]), rs(["n"], [[1]]), ordered=False)
    assert (match.ok, match.reason) == (False, "1 rows, expected 2")
    # Empty matches empty whatever the columns (a coincidence the scorer documents).
    assert compare(rs(["frame_id"], []), rs(["a", "b"], []), ordered=True).ok


def test_duplicate_values_need_a_consistent_mapping() -> None:
    expected = rs(["a", "b"], [[1, 1], [2, 2]])
    actual = rs(["x", "y", "z"], [[1, 2, 1], [2, 1, 2]])
    assert compare(expected, actual, ordered=True).ok  # a -> x, b -> z


def test_unordered_comparison_tolerates_near_equal_sort_keys() -> None:
    expected = rs(["k", "v"], [["a", 1.0000000001], ["a", 2.0]])
    actual = rs(["k", "v"], [["a", 2.0], ["a", 1.0]])
    assert compare(expected, actual, ordered=False).ok


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        (Decimal("2.5"), 2.5),
        (dt.datetime(2026, 3, 31, 15, 30, tzinfo=dt.timezone(dt.timedelta(hours=1))),
         "2026-03-31T14:30:00+00:00"),
        (dt.datetime(2026, 3, 31, 6, 0), "2026-03-31T06:00:00"),
        (dt.date(2026, 3, 31), "2026-03-31"),
        (dt.timedelta(minutes=2), 120.0),
        ("lobby", "lobby"),
        (None, None),
    ],
)  # fmt: skip
def test_normalize_value(value: Any, normalized: Any) -> None:
    assert normalize_value(value) == normalized


def test_result_set_normalizes_every_cell() -> None:
    result = result_set(["n", "at"], [(Decimal("1"), dt.datetime(2026, 3, 31, tzinfo=dt.UTC))])
    assert result.rows == [[1.0, "2026-03-31T00:00:00+00:00"]]


def test_canonical_form_sorts_unordered_rows_and_rounds_floats() -> None:
    noisy = rs(["camera", "value"], [["lobby", 0.1 + 0.2], ["entrance", 1 / 3]])
    stored = canonical(noisy, ordered=False)
    assert stored.rows == [["entrance", 0.333333333], ["lobby", 0.3]]
    assert canonical(noisy, ordered=True).rows[0][0] == "lobby"
    assert compare(stored, noisy, ordered=False).ok  # within the default tolerance
