"""Cheap per-frame input statistics for drift monitoring (Phase 9), computed on the luma plane.

- brightness: mean luma, 0..255. Reacts to darkening, glare, a covered lens.
- contrast: standard deviation of luma. Reacts to fog, haze, washed-out exposure.
- blur_var: variance of the Laplacian (a high-pass filter). Sharp edges give large values;
  defocus, motion blur or a dirty lens shrink it. Scale depends on content and resolution, so
  it is compared per camera against that camera's own reference, never across cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class InputStats:
    brightness: float
    contrast: float
    blur_var: float


def input_stats(bgr: NDArray[np.uint8]) -> InputStats:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return InputStats(
        brightness=float(mean[0, 0]),
        contrast=float(std[0, 0]),
        blur_var=float(laplacian.var()),
    )
