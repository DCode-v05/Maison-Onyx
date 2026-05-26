"""Stage 5 — Profile check.

Compares the full edge structure of the registered live piece against the
master using three metrics:
  - Hu-moment shape distance (cv2.matchShapes on the dilated edge map)
  - Edge-pixel area deviation
  - Edge IoU (dilated, so sub-pixel misalignment doesn't kill it)

The edge map is Canny over the masked grayscale image — it catches outer
outline + pave stones + cutouts + engravings, not just the silhouette.
Connected components in the XOR of the two edge maps become bounding
boxes labeled EXCESS (live has, master doesn't) or MISSING (master has,
live doesn't).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from . import visualizer
from .types import BoundingBox, ProfileCheckResult

# Thresholds — edge-based IoU is *much* lower magnitude than silhouette IoU
# because edges are sparse. These are calibration starting points; tune on a
# labeled calibration set per SKU.
SHAPE_HARD = 0.30
SHAPE_BORDER = 0.15
AREA_HARD = 0.40
AREA_BORDER = 0.20
IOU_HARD = 0.30
IOU_BORDER = 0.50
# Fraction of master edge pixels that have no near-neighbor in the live edge map.
# This is the direct answer to "how much master structure is missing on the live piece?"
MISSING_HARD = 0.40
MISSING_BORDER = 0.20

MIN_DIFF_AREA_PX = 300
EDGE_DILATE_PX = 5          # tolerance for sub-pixel alignment when comparing edge maps


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


def profile_check(
    master_bgr: np.ndarray,
    master_mask: np.ndarray,
    live_bgr: np.ndarray,
    live_mask: np.ndarray,
) -> ProfileCheckResult:
    # Full structure via Canny edges — captures internal detail (pave stones,
    # cutouts, engravings) that the segmentation mask cannot represent because
    # it gets closed into a solid blob by the morphological cleanup.
    master_edges = visualizer.edge_map(master_bgr, master_mask)
    live_edges = visualizer.edge_map(live_bgr, live_mask)

    # Dilate each edge map by EDGE_DILATE_PX. The dilated version answers
    # "is there a live (or master) edge *near* this pixel?" — the tolerance
    # that absorbs sub-pixel alignment error. The RAW edge maps are used to
    # ask the per-pixel deviation question: "is each master edge present in
    # the live image, within tolerance?"
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_DILATE_PX, EDGE_DILATE_PX))
    me_dil = cv2.dilate(master_edges, k)
    le_dil = cv2.dilate(live_edges, k)

    mm = (me_dil > 0).astype(np.uint8)
    lm = (le_dil > 0).astype(np.uint8)

    inter = (mm & lm).sum()
    union = (mm | lm).sum()
    iou = float(inter) / float(union) if union > 0 else 1.0

    a_m = int(mm.sum())
    a_l = int(lm.sum())
    if max(a_m, a_l) == 0:
        area_dev = 0.0
    else:
        area_dev = abs(a_m - a_l) / float(max(a_m, a_l))

    # matchShapes accepts a grayscale image; passing the dilated edge map
    # gives a Hu-moment distance over the full structure, not just one
    # contour.
    if a_m == 0 or a_l == 0:
        shape_dist = 1.0
    else:
        shape_dist = float(cv2.matchShapes(me_dil, le_dil, cv2.CONTOURS_MATCH_I1, 0.0))

    # Per-pixel deviation: master edge pixels NOT covered by any nearby live
    # edge. This is the direct answer to the operator's question — exactly
    # which edges from the reference are absent on the specimen.
    master_edge_bool = master_edges > 0
    live_edge_bool = live_edges > 0
    missing_edges = master_edge_bool & ~(le_dil > 0)
    excess_edges = live_edge_bool & ~(me_dil > 0)

    missing = missing_edges.astype(np.uint8) * 255
    excess = excess_edges.astype(np.uint8) * 255

    total_master_edges = int(master_edge_bool.sum())
    missing_count = int(missing_edges.sum())
    missing_edge_ratio = (
        missing_count / total_master_edges if total_master_edges > 0 else 0.0
    )

    # MISSING is the headline finding; EXCESS gets a different color label
    # so the operator can tell them apart in the per-check card.
    boxes: List[BoundingBox] = []
    boxes.extend(_box_components(missing, "MISSING", "red"))
    boxes.extend(_box_components(excess, "EXCESS", "orange"))

    # Build the deviation overlay used as the Profile cell's heatmap.
    diff_overlay = visualizer.build_profile_deviation_overlay(
        master_bgr, missing, excess
    )

    fail = (
        shape_dist > SHAPE_HARD
        or area_dev > AREA_HARD
        or iou < IOU_HARD
        or missing_edge_ratio > MISSING_HARD
    )
    border = (
        shape_dist > SHAPE_BORDER
        or area_dev > AREA_BORDER
        or iou < IOU_BORDER
        or missing_edge_ratio > MISSING_BORDER
    )
    verdict = "FAIL" if fail else ("BORDERLINE" if border else "PASS")

    return ProfileCheckResult(
        shape_distance=shape_dist,
        area_deviation=area_dev,
        silhouette_iou=iou,
        missing_edge_ratio=missing_edge_ratio,
        verdict=verdict,
        diff_overlay=diff_overlay,
        boxes=boxes,
    )
