"""Stage 3 — Geometric alignment (similarity transform).

Computes a single 2x3 affine matrix that aligns the live image to the
master's coordinate frame by simultaneously correcting:

  - translation: live centroid -> master centroid
  - rotation:    live orientation -> master orientation
  - isotropic scale: sqrt(master_area / live_area)

The matrix is applied once via cv2.warpAffine to both the live BGR
image and the live binary mask. The 180-degree orientation ambiguity
inherent to second-moment angle estimation is resolved by warping with
both candidate angles (theta, theta + 180) and keeping whichever produces
the higher masked NCC against the master.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .types import AlignmentResult


def _moments_stats(mask: np.ndarray) -> Tuple[Tuple[float, float], float, float]:
    """Return ((cx, cy), area, orientation_deg) from the binary mask."""
    m = cv2.moments(mask, binaryImage=True)
    h, w = mask.shape[:2]
    if m["m00"] <= 0:
        return (w / 2.0, h / 2.0), 0.0, 0.0
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return (float(cx), float(cy)), float(m["m00"]), float(np.degrees(theta))


def _build_similarity_matrix(
    live_center: Tuple[float, float],
    master_center: Tuple[float, float],
    theta_deg: float,
    scale: float,
) -> np.ndarray:
    """2x3 matrix that rotates+scales around live_center then translates
    so live_center lands on master_center."""
    M = cv2.getRotationMatrix2D(live_center, theta_deg, scale)
    M[0, 2] += master_center[0] - live_center[0]
    M[1, 2] += master_center[1] - live_center[1]
    return M


def _warp(image: np.ndarray, M: np.ndarray, size: Tuple[int, int],
          flags: int, border_value) -> np.ndarray:
    return cv2.warpAffine(
        image, M, size, flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def _sample_bg_color(bgr: np.ndarray) -> Tuple[int, int, int]:
    """Median of the 5-px border — used as warpAffine borderValue so that
    pixels translated/scaled outside the source extent fill with the same
    background color as the rest of the image, not black."""
    b = 5
    borders = np.concatenate([
        bgr[:b, :, :].reshape(-1, 3),
        bgr[-b:, :, :].reshape(-1, 3),
        bgr[:, :b, :].reshape(-1, 3),
        bgr[:, -b:, :].reshape(-1, 3),
    ])
    med = np.median(borders, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0
    if not np.any(sel):
        return 0.0
    av = a[sel].astype(np.float32)
    bv = b[sel].astype(np.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < 1e-6:
        return 0.0
    return float(np.dot(av, bv) / denom)


def align(
    master_bgr: np.ndarray,
    master_mask: np.ndarray,
    live_bgr: np.ndarray,
    live_mask: np.ndarray,
) -> AlignmentResult:
    h, w = live_bgr.shape[:2]

    (cx_m, cy_m), area_m, theta_m = _moments_stats(master_mask)
    (cx_l, cy_l), area_l, theta_l = _moments_stats(live_mask)

    delta_theta = theta_m - theta_l
    scale = float(np.sqrt(area_m / area_l)) if area_l > 0 else 1.0
    tx = float(cx_m - cx_l)
    ty = float(cy_m - cy_l)

    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    live_bg = _sample_bg_color(live_bgr)

    best: AlignmentResult | None = None
    for theta_candidate in (delta_theta, delta_theta + 180.0):
        M = _build_similarity_matrix(
            (cx_l, cy_l), (cx_m, cy_m), theta_candidate, scale
        )
        img = _warp(live_bgr, M, (w, h), cv2.INTER_LINEAR, live_bg)
        msk = _warp(live_mask, M, (w, h), cv2.INTER_NEAREST, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        union = ((master_mask > 0) | (msk > 0)).astype(np.uint8) * 255
        ncc = _masked_ncc(master_gray, gray, union)

        candidate = AlignmentResult(
            angle_deg=float(theta_candidate),
            scale_factor=scale,
            translation=(tx, ty),
            aligned_image=img,
            aligned_mask=msk,
            ncc=ncc,
        )
        if best is None or candidate.ncc > best.ncc:
            best = candidate

    # _moments_stats falls back to a zero-area piece with theta=0; in that
    # case both candidates produce the same NCC and the loop above still
    # populates `best`. The guard below is for type-checker peace of mind.
    assert best is not None
    return best
