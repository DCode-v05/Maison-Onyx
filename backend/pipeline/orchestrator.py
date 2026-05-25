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
    BoundingBox,
    PipelineResult,
    RegistrationResult,
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

    # Stage 1 — Preprocess (both images)
    t0 = _now()
    master_pp = preprocess.preprocess(master_input)
    live_pp = preprocess.preprocess(live_input)
    # Pad the smaller to the larger so downstream operations align in shape.
    h = max(master_pp.working.shape[0], live_pp.working.shape[0])
    w = max(master_pp.working.shape[1], live_pp.working.shape[1])

    def _pad(img: np.ndarray) -> np.ndarray:
        ph = h - img.shape[0]
        pw = w - img.shape[1]
        if ph or pw:
            return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img

    master_bgr = _pad(master_pp.working)
    live_bgr = _pad(live_pp.working)
    timings.append(StageTiming("preprocess", _ms(t0, _now())))

    # Stage 2 — Segmentation
    t0 = _now()
    master_seg = segmentation.segment(master_bgr)
    live_seg = segmentation.segment(live_bgr)
    timings.append(StageTiming("segmentation", _ms(t0, _now())))

    # Stage 3 — Rotation estimation
    t0 = _now()
    rot = rotation.estimate_rotation(
        master_bgr, master_seg.mask, live_bgr, live_seg.mask
    )
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

    # Stage 5 — Profile check
    t0 = _now()
    profile = profile_check.profile_check(master_seg.mask, live_aligned_mask)
    timings.append(StageTiming("profile_check", _ms(t0, _now())))

    # Build the shared piece silhouette once — used by both decoration and
    # surface checks to suppress anything outside the jewel.
    shared_mask = ((master_seg.mask > 0) & (live_aligned_mask > 0)).astype(np.uint8) * 255

    # Stage 6 — Decoration check
    t0 = _now()
    deco = decoration_mod.decoration_check(
        master_bgr, live_aligned, piece_mask=shared_mask
    )
    _sync()
    timings.append(StageTiming("decoration_check", _ms(t0, _now())))

    # Stage 7 — Surface check
    t0 = _now()
    surf = surface_check.surface_check(master_bgr, live_aligned, shared_mask)
    timings.append(StageTiming("surface_check", _ms(t0, _now())))

    # Stage 8 — Decision
    t0 = _now()
    decision_label, reasons = decision.decide(reg, profile, deco, surf)
    timings.append(StageTiming("decision", _ms(t0, _now())))

    # Composite difference overlay. Decoration is the only check that
    # localizes a real defect cleanly inside a patterned region — profile
    # and surface boxes are alignment noise on the per-element scale of
    # pave-set stones. The per-check cards still show what each check
    # found; the unified overlay shows only the trustworthy boxes.
    diff_boxes: List[BoundingBox] = list(deco.boxes)
    diff_overlay = visualizer.build_difference_overlay(master_bgr, live_aligned, diff_boxes)
    # Add master/live contour outlines for the profile context.
    diff_overlay = visualizer.draw_contour_outlines(
        diff_overlay, master_seg.mask, live_aligned_mask
    )

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
        live_aligned=live_aligned,
        difference_overlay=diff_overlay,
    )
