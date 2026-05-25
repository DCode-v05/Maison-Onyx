"""Stage 5 — Profile check.

Compares the silhouette of the registered live piece against the master
silhouette using three metrics:
  - Hu-moment shape distance (cv2.matchShapes)
  - Fractional area deviation
  - Silhouette IoU

Each piece passes if all three are within their thresholds. Connected
components in the XOR of the two masks become bounding boxes labeled
EXCESS (live has, master doesn't) or MISSING (master has, live doesn't).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .types import BoundingBox, ProfileCheckResult

# Thresholds — calibration starting points. Tune on a calibration set.
SHAPE_HARD = 0.10
SHAPE_BORDER = 0.05
AREA_HARD = 0.15
AREA_BORDER = 0.08
IOU_HARD = 0.85
IOU_BORDER = 0.92

MIN_DIFF_AREA_PX = 300
EROSION_PX = 7  # absorb sub-pixel misalignment between master and registered live


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _filled_silhouette(mask: np.ndarray) -> np.ndarray:
    """Return a clean filled silhouette of the largest external contour.

    The raw segmentation mask is a binarized foreground that includes interior
    holes (gaps between pave stones, open cutouts, dark shadows on the band).
    The profile check is supposed to compare outer outlines only, so we
    re-render just the largest external contour as a solid filled region.
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    contour = _largest_contour(mask)
    if contour is None:
        return out
    cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out


def _box_components(mask: np.ndarray, label: str, color: str) -> List[BoundingBox]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: List[BoundingBox] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < MIN_DIFF_AREA_PX:
            continue
        boxes.append(BoundingBox(int(x), int(y), int(w), int(h),
                                 color=color, label=label, score=float(area)))
    return boxes


def profile_check(master_mask: np.ndarray, live_mask: np.ndarray) -> ProfileCheckResult:
    # Replace the raw foreground masks with clean filled silhouettes
    # (outer contour only). Without this, every internal hole — pave gaps,
    # the open cutout, band shadows — contaminates the profile comparison.
    master_sil = _filled_silhouette(master_mask)
    live_sil = _filled_silhouette(live_mask)
    mm = (master_sil > 0).astype(np.uint8)
    lm = (live_sil > 0).astype(np.uint8)

    inter = (mm & lm).sum()
    union = (mm | lm).sum()
    iou = float(inter) / float(union) if union > 0 else 1.0

    a_m = int(mm.sum())
    a_l = int(lm.sum())
    if max(a_m, a_l) == 0:
        area_dev = 0.0
    else:
        area_dev = abs(a_m - a_l) / float(max(a_m, a_l))

    cm = _largest_contour(master_sil)
    cl = _largest_contour(live_sil)
    if cm is None or cl is None:
        shape_dist = 1.0
    else:
        shape_dist = float(cv2.matchShapes(cm, cl, cv2.CONTOURS_MATCH_I1, 0.0))

    # Diff = XOR of the filled silhouettes, eroded by EROSION_PX to absorb
    # sub-pixel registration error. Now strictly an outline comparison.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EROSION_PX, EROSION_PX))
    mm_e = cv2.erode(mm, kernel)
    lm_e = cv2.erode(lm, kernel)
    excess = (lm_e & (1 - mm_e)).astype(np.uint8) * 255   # live has, master doesn't
    missing = (mm_e & (1 - lm_e)).astype(np.uint8) * 255  # master has, live doesn't
    diff_mask = ((excess > 0) | (missing > 0)).astype(np.uint8) * 255

    boxes: List[BoundingBox] = []
    boxes.extend(_box_components(excess, "EXCESS", "red"))
    boxes.extend(_box_components(missing, "MISSING", "red"))

    fail = (shape_dist > SHAPE_HARD) or (area_dev > AREA_HARD) or (iou < IOU_HARD)
    border = (shape_dist > SHAPE_BORDER) or (area_dev > AREA_BORDER) or (iou < IOU_BORDER)
    verdict = "FAIL" if fail else ("BORDERLINE" if border else "PASS")

    return ProfileCheckResult(
        shape_distance=shape_dist,
        area_deviation=area_dev,
        silhouette_iou=iou,
        verdict=verdict,
        diff_mask=diff_mask,
        boxes=boxes,
    )
