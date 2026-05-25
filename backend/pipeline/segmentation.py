"""Stage 2 — Segmentation and centering.

Adaptive threshold + morphology + largest connected component on a
controlled (matte black / white) background. Produces a binary mask, the
bounding box of the piece, and the centroid.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .types import SegmentationResult


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold with auto polarity (handles black or white bg)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu to pick a global threshold, then decide polarity from corner brightness.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Sample the four corners: if they are mostly bright, background is white -> invert.
    h, w = gray.shape
    pad = max(2, min(h, w) // 32)
    corners = np.concatenate([
        gray[:pad, :pad].ravel(),
        gray[:pad, -pad:].ravel(),
        gray[-pad:, :pad].ravel(),
        gray[-pad:, -pad:].ravel(),
    ])
    bg_bright = float(np.mean(corners)) > 127.0
    return cv2.bitwise_not(otsu) if bg_bright else otsu


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    # skip background (label 0); pick largest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def segment(bgr: np.ndarray) -> SegmentationResult:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)

    # Morphological cleanup: close holes, then open to drop specks.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    mask = _largest_component(bw)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return SegmentationResult(
            mask=mask,
            bbox=(0, 0, w, h),
            centroid=(w / 2.0, h / 2.0),
        )

    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    M = cv2.moments(mask, binaryImage=True)
    cx = M["m10"] / M["m00"] if M["m00"] > 0 else (x0 + x1) / 2.0
    cy = M["m01"] / M["m00"] if M["m00"] > 0 else (y0 + y1) / 2.0

    return SegmentationResult(
        mask=mask,
        bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
        centroid=(float(cx), float(cy)),
    )
