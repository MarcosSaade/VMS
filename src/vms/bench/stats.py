"""Statistics shared by the benchmark tools: percentiles of samples, spread across runs."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

PERCENTILES = (50, 95, 99)


def percentiles_ms(samples_s: Sequence[float]) -> dict[str, float | None]:
    """p50/p95/p99, mean and max in milliseconds of samples in seconds; None when empty."""
    keys = [f"p{p}" for p in PERCENTILES] + ["mean", "max"]
    if not samples_s:
        return dict.fromkeys(keys)
    values = np.asarray(samples_s, dtype=np.float64) * 1000.0
    stats = [float(v) for v in np.percentile(values, PERCENTILES)]
    stats += [float(values.mean()), float(values.max())]
    return {k: round(v, 3) for k, v in zip(keys, stats, strict=True)}


def get_path(data: Mapping[str, Any], dotted: str) -> float | None:
    """The number at a dotted path (`a.b.c`), or None if it is missing or not a number."""
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def spread(values: Sequence[float]) -> dict[str, float | None]:
    """Mean, sample standard deviation (None for one value), min and max."""
    return {
        "mean": round(statistics.fmean(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else None,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def aggregate(
    reps: Sequence[Mapping[str, Any]], keys: Iterable[str]
) -> dict[str, dict[str, float | None]]:
    """`spread` of each dotted key across repetitions; keys with no numeric value are left out."""
    out: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [v for v in (get_path(r, key) for r in reps) if v is not None]
        if values:
            out[key] = spread(values)
    return out
