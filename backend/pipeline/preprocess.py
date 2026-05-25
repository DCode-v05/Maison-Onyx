"""Stage 1 — Image loading and preprocessing.

Decodes the raw image bytes (or accepts a PIL image / numpy array), and
produces a working-resolution BGR ndarray for downstream stages. Keeps the
original full-resolution copy for the surface check.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

WORKING_RESOLUTION = 1024  # square, matches the plan


@dataclass
class PreprocessedImage:
    working: np.ndarray       # BGR HxWx3 at WORKING_RESOLUTION (long side)
    full: np.ndarray          # BGR original resolution
    scale: float              # full -> working ratio


def _to_bgr_ndarray(data: Union[bytes, Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(data, bytes):
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if isinstance(data, Image.Image):
        return cv2.cvtColor(np.array(data.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        if data.shape[2] == 4:
            return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
        return data
    raise TypeError(f"Unsupported input type: {type(data)}")


def _resize_long_side(img: np.ndarray, target: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side == target:
        return img, 1.0
    scale = target / float(long_side)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, new_size, interpolation=interp), scale


def preprocess(data: Union[bytes, Image.Image, np.ndarray]) -> PreprocessedImage:
    full = _to_bgr_ndarray(data)
    working, scale = _resize_long_side(full, WORKING_RESOLUTION)
    return PreprocessedImage(working=working, full=full, scale=scale)
