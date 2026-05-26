"""Stage 3 — Rotation / similarity alignment.

Aligns the live piece to the master via a single similarity transform:

  - rotation:    live orientation -> master orientation (image moments)
  - translation: live centroid    -> master centroid
  - scale:       sqrt(master_area / live_area), isotropic

Without the translation + scale terms, the live piece stays rotated around
its own centroid — fine for orientation, but if master and live centroids
sit at different positions in their (same-size) cropped frames, every
downstream stage carries that offset. The decoration heatmap's hotspot
location is particularly sensitive to it.

The 180-deg ambiguity inherent in second-moment angle estimation is
resolved by warping with both candidates (delta, delta + 180) and keeping
whichever produces the higher masked NCC against the master.
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


def _centroid_and_area(mask: np.ndarray) -> Tuple[Tuple[float, float], float]:
    h, w = mask.shape[:2]
    m = cv2.moments(mask, binaryImage=True)
    area = float(m["m00"])
    if area <= 0:
        return (w / 2.0, h / 2.0), 0.0
    return (float(m["m10"] / area), float(m["m01"] / area)), area


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
    (live_cx, live_cy), live_area = _centroid_and_area(live_mask)
    (master_cx, master_cy), master_area = _centroid_and_area(master_mask)

    if live_area > 0 and master_area > 0:
        scale = float(np.sqrt(master_area / live_area))
    else:
        scale = 1.0

    def _transform(angle: float) -> np.ndarray:
        """Rotate around live centroid with scale, then translate the live
        centroid onto the master centroid in a single affine matrix."""
        m = cv2.getRotationMatrix2D((live_cx, live_cy), angle, scale)
        m[0, 2] += master_cx - live_cx
        m[1, 2] += master_cy - live_cy
        return m

    def _warp(img: np.ndarray, m: np.ndarray, flags: int) -> np.ndarray:
        return cv2.warpAffine(
            img, m, (w, h), flags=flags,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    m1 = _transform(delta)
    m2 = _transform(delta + 180.0)

    cand1 = _warp(live_bgr, m1, cv2.INTER_LINEAR)
    cand1_mask = _warp(live_mask, m1, cv2.INTER_NEAREST)
    cand2 = _warp(live_bgr, m2, cv2.INTER_LINEAR)
    cand2_mask = _warp(live_mask, m2, cv2.INTER_NEAREST)

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
