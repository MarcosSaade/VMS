"""Bootstrap confidence intervals for eval metrics (task 3.2).

Every quality number is a mean of per-question scores (1 correct, 0 wrong), so its uncertainty
comes from which questions happened to be in the set. The percentile bootstrap estimates it
without assuming a distribution: resample the questions with replacement, recompute the mean
each time, and take the 2.5th and 97.5th percentiles of those means as the 95% interval.

Edge cases, all deterministic for a given seed:
- no questions: no estimate (None), never 0 (an empty category is not a failing one);
- one question, or all scores equal: every resample has the same mean, so the interval has
  zero width. That reflects the data, not certainty: n is reported next to every interval, and
  a 3-question category at 100% says little.
- the seed fixes the resamples, so a rerun on the same outcomes gives the same interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Estimate(BaseModel):
    """A mean with its bootstrap interval."""

    model_config = ConfigDict(frozen=True)

    mean: float
    lo: float
    hi: float
    n: int


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default), q in [0, 1]."""
    if not sorted_values:
        raise ValueError("percentile of no values")
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def bootstrap_mean(
    values: Sequence[float], *, resamples: int, seed: int | str, confidence: float = 0.95
) -> Estimate | None:
    if not values:
        return None
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    n = len(values)
    mean = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    return Estimate(mean=mean, lo=percentile(means, tail), hi=percentile(means, 1 - tail), n=n)
