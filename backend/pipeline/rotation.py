"""Stage 3 — Rotation estimation.

Uses central moments to compute the angle of the major axis of the
segmented piece. Compares against the master angle and rotates the live
image by the delta. Resolves the 180-deg ambiguity by trying both 0 and
180 corrections and picking whichever produces the higher NCC with the
master inside the union mask.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .types import RotationResult


def _orientation_angle(mask: np.ndarray) -> float:
    """Angle of the major axis in degrees, in [-90, 90]."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return 0.0
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
    return float(np.degrees(theta))


def _rotate(img: np.ndarray, center: Tuple[float, float], angle_deg: float,
            flags: int = cv2.INTER_LINEAR, border_value=0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0
    if not np.any(sel):
        return 0.0
    av = a[sel].astype(np.float32)
    bv = b[sel].astype(np.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = (np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(av, bv) / denom)


def estimate_rotation(
    master_bgr: np.ndarray,
    master_mask: np.ndarray,
    live_bgr: np.ndarray,
    live_mask: np.ndarray,
) -> RotationResult:
    angle_master = _orientation_angle(master_mask)
    angle_live = _orientation_angle(live_mask)
    delta = angle_master - angle_live

    h, w = live_bgr.shape[:2]
    # Rotate around the centroid of the live piece.
    m = cv2.moments(live_mask, binaryImage=True)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
    else:
        cx, cy = w / 2.0, h / 2.0
    center = (cx, cy)

    # Candidate 1: delta. Candidate 2: delta + 180.
    cand1 = _rotate(live_bgr, center, delta)
    cand1_mask = _rotate(live_mask, center, delta, flags=cv2.INTER_NEAREST)

    cand2 = _rotate(live_bgr, center, delta + 180.0)
    cand2_mask = _rotate(live_mask, center, delta + 180.0, flags=cv2.INTER_NEAREST)

    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cand1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cand2, cv2.COLOR_BGR2GRAY)

    union1 = ((master_mask > 0) | (cand1_mask > 0)).astype(np.uint8) * 255
    union2 = ((master_mask > 0) | (cand2_mask > 0)).astype(np.uint8) * 255

    s1 = _ncc(master_gray, g1, union1)
    s2 = _ncc(master_gray, g2, union2)

    if s2 > s1:
        return RotationResult(angle_deg=float(delta + 180.0),
                              rotated_image=cand2,
                              rotated_mask=cand2_mask)
    return RotationResult(angle_deg=float(delta),
                          rotated_image=cand1,
                          rotated_mask=cand1_mask)
