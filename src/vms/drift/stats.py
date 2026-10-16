"""Two-sample drift statistics: a window of recent values against a reference sample.

All are pure functions of numpy arrays, shared by the live monitor (vms.drift.monitor) and by
offline scoring over stored runs (task 9.3), so both see exactly the same numbers.

- `standardized_shift`: (window mean - reference mean) / reference std, the shift of the mean
  in units of the reference's per-frame spread. Not a z-test: video frames are strongly
  autocorrelated, so a standard error of the mean would call every clean window significant.
  Its noise level on clean data is what task 9.3 calibrates its detectors against.
- `ks_statistic`: the two-sample Kolmogorov-Smirnov statistic, the largest gap between the
  two empirical CDFs, in [0, 1]. Reacts to any change of distribution, location or shape.
- `wasserstein_1`: the earth mover's distance between the two empirical distributions, in the
  values' own units: how far, on average, mass has to move. For a pure shift by d it is d.
- `psi`: the population stability index of category counts,
  sum over bins of (w - r) * ln(w / r) with w and r the bins' proportions; 0 for identical
  proportions. Conventional reading: under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 large.
  Empty bins are smoothed with `eps`, so a class that appears or vanishes counts but does not
  give infinity.

Each returns NaN when a sample is empty: an undefined statistic is reported as such, never
as "no drift".
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def standardized_shift(
    window_mean: float, reference_mean: float, reference_std: float, min_std: float
) -> float:
    """Shift of the mean in reference standard deviations; the std is floored at `min_std`
    so a signal that was constant during calibration does not divide by zero."""
    if math.isnan(window_mean) or math.isnan(reference_mean):
        return math.nan
    return (window_mean - reference_mean) / max(reference_std, min_std)


def ks_statistic(window: FloatArray, reference: FloatArray) -> float:
    if window.size == 0 or reference.size == 0:
        return math.nan
    a, b = np.sort(window), np.sort(reference)
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_1(window: FloatArray, reference: FloatArray) -> float:
    """Integral of |F_window - F_reference| over the merged support (exact for empirical
    distributions)."""
    if window.size == 0 or reference.size == 0:
        return math.nan
    a, b = np.sort(window), np.sort(reference)
    points = np.sort(np.concatenate([a, b]))
    widths = np.diff(points)
    cdf_a = np.searchsorted(a, points[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, points[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def psi(window_counts: FloatArray, reference_counts: FloatArray, eps: float = 1e-4) -> float:
    """PSI over aligned bins (same order in both arrays)."""
    if window_counts.shape != reference_counts.shape:
        raise ValueError("window and reference must have the same bins")
    if window_counts.sum() <= 0 or reference_counts.sum() <= 0:
        return math.nan
    w = np.maximum(window_counts / window_counts.sum(), eps)
    r = np.maximum(reference_counts / reference_counts.sum(), eps)
    w, r = w / w.sum(), r / r.sum()
    return float(np.sum((w - r) * np.log(w / r)))
