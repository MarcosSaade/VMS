"""Bootstrap intervals: deterministic, sensible width, and the documented edge cases."""

from __future__ import annotations

import pytest

from vms.eval.stats import bootstrap_mean, percentile


def test_percentile_interpolates_like_numpy() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    with pytest.raises(ValueError, match="no values"):
        percentile([], 0.5)


def test_interval_contains_the_mean_and_is_reproducible() -> None:
    values = [1.0] * 70 + [0.0] * 30
    a = bootstrap_mean(values, resamples=2000, seed="s")
    assert a is not None
    assert a.mean == 0.7
    assert a.lo < 0.7 < a.hi
    assert a == bootstrap_mean(values, resamples=2000, seed="s")
    assert a != bootstrap_mean(values, resamples=2000, seed="other")
    # Normal approximation: +/- 1.96 * sqrt(0.21 / 100) = +/- 0.09.
    assert 0.07 < a.hi - 0.7 < 0.11
    assert 0.07 < 0.7 - a.lo < 0.11


def test_interval_narrows_with_more_questions() -> None:
    small = bootstrap_mean([1.0, 0.0] * 10, resamples=2000, seed=1)
    large = bootstrap_mean([1.0, 0.0] * 200, resamples=2000, seed=1)
    assert small is not None
    assert large is not None
    assert (large.hi - large.lo) < (small.hi - small.lo) / 3


def test_edge_cases() -> None:
    assert bootstrap_mean([], resamples=2000, seed=1) is None
    one = bootstrap_mean([1.0], resamples=2000, seed=1)
    assert one is not None
    assert (one.mean, one.lo, one.hi, one.n) == (1.0, 1.0, 1.0, 1)
    same = bootstrap_mean([0.0] * 30, resamples=2000, seed=1)
    assert same is not None
    assert (same.lo, same.hi) == (0.0, 0.0)
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_mean([1.0], resamples=0, seed=1)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean([1.0], resamples=10, seed=1, confidence=1.0)


def test_wider_confidence_gives_a_wider_interval() -> None:
    values = [1.0] * 60 + [0.0] * 40
    narrow = bootstrap_mean(values, resamples=2000, seed=3, confidence=0.8)
    wide = bootstrap_mean(values, resamples=2000, seed=3, confidence=0.99)
    assert narrow is not None
    assert wide is not None
    assert wide.lo < narrow.lo
    assert wide.hi > narrow.hi
