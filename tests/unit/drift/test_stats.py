"""Drift statistics on synthetic distributions with known shifts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vms.drift.stats import ks_statistic, psi, standardized_shift, wasserstein_1

RNG_SEED = 20261016


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def test_standardized_shift_is_in_reference_stds() -> None:
    assert standardized_shift(110.0, 100.0, 5.0, min_std=1.0) == pytest.approx(2.0)
    assert standardized_shift(90.0, 100.0, 5.0, min_std=1.0) == pytest.approx(-2.0)
    # A constant reference uses the floor instead of dividing by zero.
    assert standardized_shift(1.0, 0.0, 0.0, min_std=0.5) == pytest.approx(2.0)
    assert math.isnan(standardized_shift(math.nan, 0.0, 1.0, min_std=0.5))


def test_ks_is_zero_for_identical_samples_and_one_for_disjoint() -> None:
    a = np.array([0.1, 0.4, 0.4, 0.9])
    assert ks_statistic(a, a.copy()) == 0.0
    assert ks_statistic(a, a + 10) == 1.0


def test_ks_matches_the_population_value_of_a_normal_shift(rng: np.random.Generator) -> None:
    # Two-sample KS of N(0, 1) against N(d, 1) tends to 2 Phi(d / 2) - 1.
    reference = rng.normal(0, 1, 20_000)
    for d in (0.0, 0.5, 1.0, 2.0):
        expected = 2 * normal_cdf(d / 2) - 1
        assert ks_statistic(rng.normal(d, 1, 20_000), reference) == pytest.approx(
            expected, abs=0.02
        )


def test_ks_reacts_to_shape_with_the_same_mean(rng: np.random.Generator) -> None:
    reference = rng.normal(0.5, 0.05, 5000)
    wider = rng.normal(0.5, 0.2, 5000)
    assert ks_statistic(reference[:2500], reference[2500:]) < 0.05
    # Population value: max over t of Phi(t) - Phi(t / 4) = 0.291.
    assert ks_statistic(wider, reference) == pytest.approx(0.291, abs=0.03)


def test_wasserstein_of_a_pure_shift_is_the_shift(rng: np.random.Generator) -> None:
    a = rng.uniform(0, 1, 1000)
    assert wasserstein_1(a + 0.3, a) == pytest.approx(0.3)
    assert wasserstein_1(a, a.copy()) == pytest.approx(0.0)


def test_wasserstein_of_point_masses_and_unequal_sizes() -> None:
    assert wasserstein_1(np.array([0.0]), np.array([2.0])) == pytest.approx(2.0)
    # Half the mass moves 1: {0, 1} vs {0, 0, 0, 0}.
    assert wasserstein_1(np.array([0.0, 1.0]), np.zeros(4)) == pytest.approx(0.5)


def test_psi_known_values() -> None:
    same = np.array([50.0, 30.0, 20.0])
    assert psi(same, same * 3) == pytest.approx(0.0)
    window, reference = np.array([20.0, 80.0]), np.array([50.0, 50.0])
    expected = (0.2 - 0.5) * math.log(0.2 / 0.5) + (0.8 - 0.5) * math.log(0.8 / 0.5)
    assert psi(window, reference) == pytest.approx(expected)


def test_psi_of_a_new_class_is_large_and_finite() -> None:
    reference = np.array([100.0, 0.0])
    window = np.array([50.0, 50.0])
    value = psi(window, reference)
    assert math.isfinite(value)
    assert value > 1.0


def test_psi_requires_aligned_bins() -> None:
    with pytest.raises(ValueError, match="same bins"):
        psi(np.ones(2), np.ones(3))


@pytest.mark.parametrize("stat", [ks_statistic, wasserstein_1])
def test_empty_samples_are_undefined(stat: object) -> None:
    fn = stat  # type: ignore[assignment]
    assert math.isnan(fn(np.array([]), np.array([1.0])))  # type: ignore[operator]
    assert math.isnan(fn(np.array([1.0]), np.array([])))  # type: ignore[operator]


def test_psi_of_no_counts_is_undefined() -> None:
    assert math.isnan(psi(np.zeros(3), np.ones(3)))
