"""Image transforms that simulate camera drift (task 9.2), shared by loadgen's drift injection
and the detection eval under drift, so both apply exactly the same pixels.

Each transform takes a BGR image and an intensity in [0, 1] (0 leaves the image untouched; a
gradual drift ramps it up) and is defined by one `strength` at full intensity:

- darken: multiply every pixel by a gain going from 1 to `strength` (e.g. 0.3). Dusk, a
  failed light, an iris stuck closed. Photometric: boxes stay valid.
- blur: Gaussian blur with sigma up to `strength` pixels at 1280 px width (scaled with the
  image width, so a COCO image and a camera frame blur alike relative to their size).
  Defocus, a wet or dirty lens. Photometric.
- noise: additive Gaussian noise with sigma up to `strength` levels, independent per pixel and
  colour channel, drawn per frame from the given seeded generator (in luma, which mixes the
  channels, its sigma is about 0.67 of that). Sensor gain at night, interference. Photometric.
- shift: the camera turns: the view is cropped to (1 - strength) of each side, offset toward
  the bottom right, and scaled back to full size, so the scene moves and zooms. Geometric:
  boxes move (the detection eval skips it).
- occlusion: a black rectangle covering up to `strength` of the frame area, at a position
  fixed per camera (`anchor`, from a seeded generator): a parked truck, a sign, a spider web.
  Boxes stay where they are, but objects behind the rectangle vanish.

The class-mix drift is not a pixel transform: loadgen switches the camera to another clip.
"""

from __future__ import annotations

import math
from typing import Final, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
TransformKind = Literal["darken", "blur", "noise", "shift", "occlusion"]
PHOTOMETRIC: Final = ("darken", "blur", "noise")
REFERENCE_WIDTH: Final = 1280


def check_strength(kind: TransformKind, strength: float) -> None:
    ok = {
        "darken": 0 <= strength < 1,
        "blur": strength > 0,
        "noise": strength > 0,
        "shift": 0 < strength < 0.5,
        "occlusion": 0 < strength < 1,
    }[kind]
    if not ok:
        raise ValueError(f"strength {strength} out of range for {kind}")


def apply(
    kind: TransformKind,
    image: Image,
    intensity: float,
    strength: float,
    *,
    rng: np.random.Generator | None = None,
    anchor: tuple[float, float] = (0.5, 0.5),
) -> Image:
    """`image` transformed at `intensity` (clipped to [0, 1]); a new array unless intensity
    is 0. `rng` is needed for noise; `anchor` (in [0, 1]^2) places the occluder."""
    s = min(max(intensity, 0.0), 1.0)
    if s == 0.0:
        return image
    if kind == "darken":
        return _darken(image, 1.0 - s * (1.0 - strength))
    if kind == "blur":
        return _blur(image, s * strength)
    if kind == "noise":
        if rng is None:
            raise ValueError("noise needs a seeded generator")
        return _noise(image, s * strength, rng)
    if kind == "shift":
        return _shift(image, s * strength)
    return _occlude(image, s * strength, anchor)


def _darken(image: Image, gain: float) -> Image:
    return cast(Image, cv2.convertScaleAbs(image, alpha=gain, beta=0))


def _blur(image: Image, sigma_ref: float) -> Image:
    sigma = sigma_ref * image.shape[1] / REFERENCE_WIDTH
    if sigma < 0.05:
        return image
    return cast(Image, cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma))


def _noise(image: Image, sigma: float, rng: np.random.Generator) -> Image:
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).round().astype(np.uint8)


def _shift(image: Image, fraction: float) -> Image:
    h, w = image.shape[:2]
    dx, dy = round(w * fraction), round(h * fraction)
    if dx == 0 and dy == 0:
        return image
    crop = image[dy:, dx:]
    return cast(Image, cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR))


def _occlude(image: Image, area: float, anchor: tuple[float, float]) -> Image:
    """A rectangle with the frame's aspect ratio covering `area` of it, centred at `anchor`
    and kept inside the frame."""
    h, w = image.shape[:2]
    side = math.sqrt(area)
    rw, rh = round(w * side), round(h * side)
    if rw == 0 or rh == 0:
        return image
    x0 = min(max(round(anchor[0] * w - rw / 2), 0), w - rw)
    y0 = min(max(round(anchor[1] * h - rh / 2), 0), h - rh)
    out = image.copy()
    out[y0 : y0 + rh, x0 : x0 + rw] = 0
    return out
