"""Cheap per-frame input statistics for drift monitoring (Phase 9), computed on the luma plane.

- brightness: mean luma, 0..255. Reacts to darkening, glare, a covered lens.
- contrast: standard deviation of luma. Reacts to fog, haze, washed-out exposure.
- blur_var: variance of the Laplacian (a high-pass filter). Sharp edges give large values;
  defocus, motion blur or a dirty lens shrink it. Scale depends on content and resolution, so
  it is compared per camera against that camera's own reference, never across cameras.
- noise: estimated standard deviation of additive Gaussian noise, in luma levels (Immerkaer,
  "Fast noise variance estimation", 1996): the absolute response of a 3x3 mask that cancels
  any locally planar image, scaled so pure N(0, sigma^2) noise gives sigma. Strong edges and
  fine texture add to it, and JPEG compression removes some noise, so like blur_var it is a
  per-camera signal: what matters is its change against the camera's reference (sensor gain
  at night, a failing sensor, interference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class InputStats:
    brightness: float
    contrast: float
    blur_var: float
    noise: float


# Sum of two Laplacian-like masks: zero response on any locally planar patch.
_NOISE_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_NOISE_SCALE = math.sqrt(math.pi / 2) / 6  # E|mask * noise| = 6 sigma sqrt(2 / pi)


def input_stats(bgr: NDArray[np.uint8]) -> InputStats:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return InputStats(
        brightness=float(mean[0, 0]),
        contrast=float(std[0, 0]),
        blur_var=float(laplacian.var()),
        noise=noise_sigma(gray),
    )


def noise_sigma(gray: NDArray[Any]) -> float:
    """Immerkaer's estimate of the noise standard deviation of a grayscale image (border
    pixels excluded); 0 for images smaller than 3x3."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    response = cv2.filter2D(gray.astype(np.float32), -1, _NOISE_MASK)[1:-1, 1:-1]
    return float(_NOISE_SCALE * np.abs(response).mean())
