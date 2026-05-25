"""Stage 4 — Fine registration with SIFT.

Detects SIFT keypoints in master (precomputed) and the coarse-aligned
live image. Matches with Lowe's ratio test and computes a homography via
RANSAC. Warps the live image into the master frame. Reports NCC,
inlier count, and a reliability flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import RegistrationResult

MAX_KEYPOINTS = 600
RATIO = 0.7
MIN_INLIERS = 15


@dataclass
class MasterFeatures:
    keypoints: Tuple
    descriptors: Optional[np.ndarray]


_sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS)
# FLANN kd-tree matcher — typically 2-5x faster than brute force on SIFT (128-d).
_FLANN_INDEX_KDTREE = 1
_matcher = cv2.FlannBasedMatcher(
    dict(algorithm=_FLANN_INDEX_KDTREE, trees=4),
    dict(checks=32),
)


def compute_master_features(master_gray: np.ndarray, master_mask: np.ndarray) -> MasterFeatures:
    kps, desc = _sift.detectAndCompute(master_gray, master_mask)
    return MasterFeatures(keypoints=kps, descriptors=desc)


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Public wrapper so the orchestrator can decide whether SIFT is needed."""
    return _ncc(a, b, mask)


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


def register(
    master_bgr: np.ndarray,
    master_mask: np.ndarray,
    live_bgr: np.ndarray,
    live_mask: np.ndarray,
    master_features: Optional[MasterFeatures] = None,
) -> RegistrationResult:
    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY)

    if master_features is None:
        master_features = compute_master_features(master_gray, master_mask)
    live_kp, live_desc = _sift.detectAndCompute(live_gray, live_mask)

    h, w = master_bgr.shape[:2]
    fallback = RegistrationResult(
        warped=live_bgr.copy(),
        warped_mask=live_mask.copy(),
        homography=None,
        num_inliers=0,
        inlier_ratio=0.0,
        ncc=_ncc(master_gray, live_gray, ((master_mask > 0) | (live_mask > 0)).astype(np.uint8) * 255),
        reliable=False,
    )

    if master_features.descriptors is None or live_desc is None:
        return fallback
    if len(master_features.descriptors) < 4 or len(live_desc) < 4:
        return fallback

    knn = _matcher.knnMatch(live_desc, master_features.descriptors, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good.append(m)
    if len(good) < 4:
        return fallback

    src = np.float32([live_kp[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([master_features.keypoints[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inlier_mask is None:
        return fallback

    num_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_inliers) / float(len(good))

    warped = cv2.warpPerspective(live_bgr, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_mask = cv2.warpPerspective(live_mask, H, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    union = ((master_mask > 0) | (warped_mask > 0)).astype(np.uint8) * 255
    ncc = _ncc(master_gray, warped_gray, union)

    reliable = num_inliers >= MIN_INLIERS and inlier_ratio >= 0.3

    return RegistrationResult(
        warped=warped,
        warped_mask=warped_mask,
        homography=H,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        ncc=ncc,
        reliable=reliable,
    )
