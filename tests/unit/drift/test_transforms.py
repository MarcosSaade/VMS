"""Drift transforms: each changes what it should, by the amount its strength says."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vms.drift.transforms import apply, check_strength
from vms.worker.stats import input_stats


def scene(width: int = 320, height: int = 240) -> np.ndarray:
    """A textured image: a ramp plus seeded blobs, no saturated pixels."""
    rng = np.random.default_rng(5)
    ramp = np.tile(np.linspace(60, 180, width), (height, 1))
    blobs = cv2.GaussianBlur(rng.uniform(-40, 40, (height, width)), (0, 0), 0.7)
    gray = np.clip(ramp + blobs, 0, 255).astype(np.uint8)
    return np.dstack([gray] * 3)


def test_zero_intensity_is_the_identity() -> None:
    image = scene()
    assert apply("darken", image, 0.0, 0.3) is image
    assert apply("noise", image, -1.0, 10.0) is image  # clipped to 0


def test_darken_scales_brightness_by_the_gain() -> None:
    image = scene()
    base = input_stats(image).brightness
    assert input_stats(apply("darken", image, 1.0, 0.3)).brightness == pytest.approx(
        0.3 * base, rel=0.02
    )
    # Half way through a ramp: gain 1 - 0.5 * 0.7 = 0.65.
    assert input_stats(apply("darken", image, 0.5, 0.3)).brightness == pytest.approx(
        0.65 * base, rel=0.02
    )


def test_blur_lowers_sharpness_and_scales_with_width() -> None:
    small, large = scene(320, 240), scene(1280, 960)
    for image in (small, large):
        assert input_stats(apply("blur", image, 1.0, 4.0)).blur_var < (
            0.2 * input_stats(image).blur_var
        )
    # Sigma is given at 1280 px: 4 there, 1 at 320 px, 2 half way through a ramp.
    expected = cv2.GaussianBlur(small, (0, 0), sigmaX=1.0, sigmaY=1.0)
    assert np.array_equal(apply("blur", small, 1.0, 4.0), expected)
    expected = cv2.GaussianBlur(large, (0, 0), sigmaX=2.0, sigmaY=2.0)
    assert np.array_equal(apply("blur", large, 0.5, 4.0), expected)


def test_noise_has_the_requested_sigma_and_is_seeded() -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    out = apply("noise", image, 1.0, 12.0, rng=np.random.default_rng(1))
    assert out.astype(float).std() == pytest.approx(12.0, rel=0.03)
    again = apply("noise", image, 1.0, 12.0, rng=np.random.default_rng(1))
    assert np.array_equal(out, again)
    # Independent noise per channel: luma averages it down by sqrt(0.299^2 + 0.587^2 +
    # 0.114^2) = 0.668.
    assert input_stats(out).noise == pytest.approx(12.0 * 0.668, rel=0.05)
    with pytest.raises(ValueError, match="seeded"):
        apply("noise", image, 1.0, 12.0)


def test_shift_moves_the_scene_and_keeps_the_size() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[40:60, 90:110] = 255  # a square at the centre
    out = apply("shift", image, 1.0, 0.2)
    assert out.shape == image.shape
    ys, xs = np.nonzero(out[..., 0] > 127)
    # Cropped by 40 px left and 20 px top, then scaled by 1 / 0.8.
    assert xs.mean() == pytest.approx((100 - 40) / 0.8, abs=2)
    assert ys.mean() == pytest.approx((50 - 20) / 0.8, abs=2)


def test_occlusion_blacks_out_the_area_inside_the_frame() -> None:
    image = np.full((90, 160, 3), 200, dtype=np.uint8)
    out = apply("occlusion", image, 1.0, 0.25, anchor=(0.95, 0.95))  # pushed into the frame
    black = (out[..., 0] == 0).mean()
    assert black == pytest.approx(0.25, abs=0.01)
    assert out[-1, -1, 0] == 0  # clamped into the corner
    assert (image == 200).all()  # the input is untouched


@pytest.mark.parametrize(
    ("kind", "strength"),
    [("darken", 1.0), ("blur", 0.0), ("noise", -1.0), ("shift", 0.5), ("occlusion", 1.0)],
)
def test_strength_ranges(kind: str, strength: float) -> None:
    with pytest.raises(ValueError, match="out of range"):
        check_strength(kind, strength)  # type: ignore[arg-type]
