"""Shared data types for the inspection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class StageTiming:
    name: str
    ms: float


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    color: str          # "red" | "orange" | "magenta"
    label: str
    score: Optional[float] = None


@dataclass
class SegmentationResult:
    mask: np.ndarray              # uint8 0/255
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]


@dataclass
class RotationResult:
    angle_deg: float
    rotated_image: np.ndarray
    rotated_mask: np.ndarray


@dataclass
class RegistrationResult:
    warped: np.ndarray
    warped_mask: np.ndarray
    homography: Optional[np.ndarray]
    num_inliers: int
    inlier_ratio: float
    ncc: float
    reliable: bool


@dataclass
class ProfileCheckResult:
    shape_distance: float
    area_deviation: float
    silhouette_iou: float          # actually edge IoU since the switch to Canny
    missing_edge_ratio: float      # fraction of master edges absent in live
    verdict: str                   # PASS | BORDERLINE | FAIL
    diff_overlay: np.ndarray       # BGR — master dimmed + red missing + orange excess
    boxes: List[BoundingBox] = field(default_factory=list)


@dataclass
class DecorationCheckResult:
    global_similarity: float
    problem_patch_ratio: float
    max_color_distance: float    # LAB delta-E worst-case across all patches
    heatmap: np.ndarray      # H x W float32 in [0, 1]
    diff_overlay: np.ndarray # BGR — JET heatmap blended over live + bboxes drawn
    verdict: str
    boxes: List[BoundingBox] = field(default_factory=list)


@dataclass
class SurfaceCheckResult:
    defect_ratio: float
    num_defect_regions: int
    max_defect_size: int
    defect_map: np.ndarray   # H x W float32 in [0, 1]
    verdict: str
    boxes: List[BoundingBox] = field(default_factory=list)


@dataclass
class PipelineResult:
    decision: str            # ACCEPT | REVIEW | REJECT
    reasons: List[str]
    profile: ProfileCheckResult
    decoration: DecorationCheckResult
    surface: SurfaceCheckResult
    registration: RegistrationResult
    rotation: RotationResult
    timings: List[StageTiming]
    total_ms: float

    # Master panel
    master_image: np.ndarray          # BGR — bare reference at working resolution
    master_contoured: np.ndarray      # BGR — external + internal contours drawn over master

    # Live stage progression (left -> right in the UI)
    live_preprocessed: np.ndarray     # after Stage 1
    live_segmented: np.ndarray        # after Stage 2 — mask overlay on live
    live_rotated: np.ndarray          # after Stage 3 — moment-based rotation
    live_registered: np.ndarray       # after Stage 4 — SIFT-aligned, with contours overlaid
