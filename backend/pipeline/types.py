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
    silhouette_iou: float
    verdict: str            # PASS | BORDERLINE | FAIL
    diff_mask: np.ndarray
    boxes: List[BoundingBox] = field(default_factory=list)


@dataclass
class DecorationCheckResult:
    global_similarity: float
    problem_patch_ratio: float
    heatmap: np.ndarray      # H x W float32 in [0, 1]
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
    master_image: np.ndarray          # BGR
    live_aligned: np.ndarray          # BGR (post-registration)
    difference_overlay: np.ndarray    # BGR with all boxes drawn
