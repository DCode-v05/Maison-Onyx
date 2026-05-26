"""Pipeline orchestrator — runs all 8 stages, in order, with per-stage timing."""

from __future__ import annotations

import time
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from . import decoration_check as decoration_mod
from . import registration as registration_mod
from . import (
    decision,
    preprocess,
    profile_check,
    rotation,
    segmentation,
    surface_check,
    visualizer,
)
from .types import (
    PipelineResult,
    RegistrationResult,
    SegmentationResult,
    StageTiming,
)

# If rotation already aligns the live well enough, skip SIFT entirely.
ROTATION_GOOD_ENOUGH_NCC = 0.92


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    return time.perf_counter()


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _seg_from_mask(mask: np.ndarray) -> SegmentationResult:
    """Rebuild a SegmentationResult from a cropped mask.

    After the tight-bbox crop, the original SegmentationResult (which referred
    to the full frame's coordinate space) is stale. This recomputes bbox and
    centroid relative to the cropped mask so rotation, registration, and the
    three checks all operate in a consistent frame.
    """
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return SegmentationResult(
            mask=mask, bbox=(0, 0, w, h), centroid=(w / 2.0, h / 2.0)
        )
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    m = cv2.moments(mask, binaryImage=True)
    cx = m["m10"] / m["m00"] if m["m00"] > 0 else (x0 + x1) / 2.0
    cy = m["m01"] / m["m00"] if m["m00"] > 0 else (y0 + y1) / 2.0
    return SegmentationResult(
        mask=mask,
        bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
        centroid=(float(cx), float(cy)),
    )


def warmup(n: int = 3) -> None:
    """Hit each model path once so first real request is not cold-start."""
    decoration_mod.init_model()
    dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(dummy, cv2.COLOR_RGB2BGR)
    for _ in range(n):
        decoration_mod.extract_features(bgr)
        _sync()


def run_pipeline(
    master_input: Union[bytes, Image.Image, np.ndarray],
    live_input: Union[bytes, Image.Image, np.ndarray],
) -> PipelineResult:
    timings: List[StageTiming] = []

    # Stage 1 — Preprocess. Resizes each input so its long side is the
    # working resolution. No padding to a common shape — each image keeps
    # its native aspect ratio; the tight-bbox crop in Stage 2 makes them
    # comparable.
    t0 = _now()
    master_pp = preprocess.preprocess(master_input)
    live_pp = preprocess.preprocess(live_input)
    master_full = master_pp.working
    live_full = live_pp.working
    live_preprocessed = live_full.copy()
    timings.append(StageTiming("preprocess", _ms(t0, _now())))

    # Stage 2 — Segmentation + tight bounding-box crop.
    # Each piece is cropped to its own (xmin, ymin, xmax, ymax) from
    # segmentation — no extra padding around the silhouette. Live is then
    # resized to master's HxW so downstream stages compare like-for-like in
    # the master's coordinate frame.
    t0 = _now()
    master_seg_full = segmentation.segment(master_full)
    live_seg_full = segmentation.segment(live_full)

    mx, my, mw, mh = master_seg_full.bbox
    lx, ly, lw, lh = live_seg_full.bbox

    master_bgr = master_full[my:my + mh, mx:mx + mw].copy()
    master_mask = master_seg_full.mask[my:my + mh, mx:mx + mw].copy()

    live_crop_bgr = live_full[ly:ly + lh, lx:lx + lw].copy()
    live_crop_mask = live_seg_full.mask[ly:ly + lh, lx:lx + lw].copy()

    if (lh, lw) != (mh, mw):
        live_bgr = cv2.resize(live_crop_bgr, (mw, mh), interpolation=cv2.INTER_LINEAR)
        live_mask = cv2.resize(live_crop_mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
    else:
        live_bgr = live_crop_bgr
        live_mask = live_crop_mask

    master_seg = _seg_from_mask(master_mask)
    live_seg = _seg_from_mask(live_mask)

    live_segmented = visualizer.overlay_mask(live_bgr, live_seg.mask)
    timings.append(StageTiming("segmentation", _ms(t0, _now())))

    # Stage 3 — Rotation estimation
    t0 = _now()
    rot = rotation.estimate_rotation(
        master_bgr, master_seg.mask, live_bgr, live_seg.mask
    )
    live_rotated = rot.rotated_image.copy()
    timings.append(StageTiming("rotation_estimation", _ms(t0, _now())))

    # Stage 4 — Fine registration. Skip SIFT if rotation already aligned the
    # piece well — no point spending ~1 s producing a near-identity homography.
    t0 = _now()
    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    rotated_gray = cv2.cvtColor(rot.rotated_image, cv2.COLOR_BGR2GRAY)
    union_mask = ((master_seg.mask > 0) | (rot.rotated_mask > 0)).astype(np.uint8) * 255
    rot_ncc = registration_mod.masked_ncc(master_gray, rotated_gray, union_mask)

    if rot_ncc >= ROTATION_GOOD_ENOUGH_NCC:
        reg = RegistrationResult(
            warped=rot.rotated_image,
            warped_mask=rot.rotated_mask,
            homography=np.eye(3),
            num_inliers=0,
            inlier_ratio=1.0,
            ncc=rot_ncc,
            reliable=True,
        )
    else:
        reg = registration_mod.register(
            master_bgr, master_seg.mask, rot.rotated_image, rot.rotated_mask
        )
    timings.append(StageTiming("fine_registration", _ms(t0, _now())))

    # Use registered live throughout; fall back to rotated if registration unreliable.
    live_aligned = reg.warped
    live_aligned_mask = reg.warped_mask

    # Stage 5 — Profile check (edge-based; needs the BGR images, not just masks)
    t0 = _now()
    profile = profile_check.profile_check(
        master_bgr, master_seg.mask, live_aligned, live_aligned_mask
    )
    timings.append(StageTiming("profile_check", _ms(t0, _now())))

    # Stage 6 — Decoration check.
    # Compares the segmented master against the rotation-aligned live (NOT
    # the SIFT-warped live). Working on the rotation-aligned frame avoids
    # any homography-induced texture warping that could shift DINOv2 patch
    # features and produce false decoration deviations.
    t0 = _now()
    deco_shared_mask = ((master_seg.mask > 0) & (rot.rotated_mask > 0)).astype(np.uint8) * 255
    deco = decoration_mod.decoration_check(
        master_bgr, rot.rotated_image, piece_mask=deco_shared_mask
    )
    _sync()
    timings.append(StageTiming("decoration_check", _ms(t0, _now())))

    # Stage 7 — Surface check (SIFT-aligned live; needs sub-pixel registration).
    shared_mask = ((master_seg.mask > 0) & (live_aligned_mask > 0)).astype(np.uint8) * 255
    t0 = _now()
    surf = surface_check.surface_check(master_bgr, live_aligned, shared_mask)
    timings.append(StageTiming("surface_check", _ms(t0, _now())))

    # Stage 8 — Decision
    t0 = _now()
    decision_label, reasons = decision.decide(reg, profile, deco, surf)
    timings.append(StageTiming("decision", _ms(t0, _now())))

    # Build the edge overlays — these mirror what the profile check actually
    # compares (Canny edges over the masked grayscale). Drawing them on the
    # master and the live registered image makes the "what was aligned, and
    # where does its structure differ" answerable at a glance.
    master_edges = visualizer.edge_map(master_bgr, master_seg.mask)
    live_edges = visualizer.edge_map(live_aligned, live_aligned_mask)
    master_contoured = visualizer.draw_edges(master_bgr, master_edges)
    live_registered = visualizer.draw_edges(live_aligned, live_edges)

    total_ms = sum(s.ms for s in timings)
    timings.append(StageTiming("total", total_ms))

    return PipelineResult(
        decision=decision_label,
        reasons=reasons,
        profile=profile,
        decoration=deco,
        surface=surf,
        registration=reg,
        rotation=rot,
        timings=timings,
        total_ms=total_ms,
        master_image=master_bgr,
        master_contoured=master_contoured,
        live_preprocessed=live_preprocessed,
        live_segmented=live_segmented,
        live_rotated=live_rotated,
        live_registered=live_registered,
    )
