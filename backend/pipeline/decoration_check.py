"""Stage 6 — Decoration check using DINOv2 patch features.

Extracts per-patch features from the master and the registered live image
using DINOv2 ViT-S/14 (or B/14). Computes cosine similarity per patch
location and a global similarity. Patches below threshold form low-similarity
regions; their connected components become bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from . import visualizer
from .types import BoundingBox, DecorationCheckResult

# ViT-S/14 by default per the plan; flip to "facebook/dinov2-base" if needed.
DEFAULT_MODEL_NAME = "facebook/dinov2-small"
PATCH = 14
INPUT_SIZE = 224          # standard DINOv2 input
PATCHES_PER_SIDE = INPUT_SIZE // PATCH  # 16

GLOBAL_HARD = 0.85
GLOBAL_BORDER = 0.90
RATIO_HARD = 0.10
RATIO_BORDER = 0.04
PATCH_SIM_THRESHOLD = 0.75

# A single confidently flagged region (e.g. one missing pave stone) barely
# moves the global average on a piece with 20+ stones. Escalate by box area
# so the verdict still catches localized decoration defects.
LOCAL_DEFECT_BORDER_PX = 500
LOCAL_DEFECT_HARD_PX = 2000

# Color deviation — Euclidean distance in CIELAB space (roughly delta-E).
# Delta-E ≈ 2.3 is the "just noticeable difference"; 12 is "clearly different";
# 25+ is "different color class". Per-patch comparison; matches DINOv2's grid.
COLOR_DIST_HARD = 25.0      # clearly different color → definite defect
COLOR_DIST_BORDER = 12.0    # noticeable shift → review
MAX_USEFUL_COLOR_DIST = 50.0  # for normalizing color deviation to [0, 1]


@dataclass
class DinoFeatures:
    patch_tokens: np.ndarray         # (P*P, D) L2-normalized
    grid_size: int                    # P


@dataclass
class _Holder:
    model: Optional[AutoModel] = None
    processor: Optional[AutoImageProcessor] = None
    device: str = "cpu"
    name: str = DEFAULT_MODEL_NAME


_h = _Holder()


def init_model(model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _h.model is not None and _h.name == model_name and _h.device == device:
        return
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    _h.model = model
    _h.processor = proc
    _h.device = device
    _h.name = model_name


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@torch.inference_mode()
def extract_features(bgr: np.ndarray) -> DinoFeatures:
    if _h.model is None:
        init_model()
    pil = _to_pil(bgr)
    inputs = _h.processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(_h.device) for k, v in inputs.items()}
    out = _h.model(**inputs)
    # last_hidden_state: (1, 1+P*P, D) - drop CLS
    tokens = out.last_hidden_state[0, 1:, :]
    tokens = torch.nn.functional.normalize(tokens, p=2, dim=1)
    arr = tokens.cpu().numpy()
    grid = int(np.sqrt(arr.shape[0]))
    return DinoFeatures(patch_tokens=arr, grid_size=grid)


def _box_components(mask: np.ndarray, sim_map: np.ndarray) -> list[BoundingBox]:
    # Bridge small gaps between low-sim pixels of the same defect so we draw
    # ONE box per missing element instead of a scatter of slivers.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    num, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes: list[BoundingBox] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 250:
            continue
        roi = sim_map[y:y + h, x:x + w]
        score = float(roi[roi < PATCH_SIM_THRESHOLD].mean()) if np.any(roi < PATCH_SIM_THRESHOLD) else float(roi.mean())
        boxes.append(BoundingBox(int(x), int(y), int(w), int(h),
                                 color="orange",
                                 label=f"Sim: {score:.2f}",
                                 score=score))
    return boxes


def _color_box_components(mask: np.ndarray, color_dist_map: np.ndarray) -> list[BoundingBox]:
    """Bounding boxes for color-defect regions. Magenta to distinguish from
    DINOv2 structural-defect boxes (orange)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    num, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes: list[BoundingBox] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 150:
            continue
        roi = color_dist_map[y:y + h, x:x + w]
        score = float(roi[roi > COLOR_DIST_BORDER].mean()) if np.any(roi > COLOR_DIST_BORDER) else float(roi.mean())
        boxes.append(BoundingBox(int(x), int(y), int(w), int(h),
                                 color="magenta",
                                 label=f"Color ΔE: {score:.0f}",
                                 score=score))
    return boxes


def _patch_mean_lab(lab: np.ndarray, grid_size: int, piece_mask: Optional[np.ndarray]) -> np.ndarray:
    """Mean LAB color per grid cell. Returns (grid_size, grid_size, 3) float32.

    Patches that are entirely outside the piece mask return (0, 0, 0) — they
    won't get compared (the silhouette gate downstream filters them out).
    """
    h, w = lab.shape[:2]
    out = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    for gy in range(grid_size):
        y0 = (gy * h) // grid_size
        y1 = ((gy + 1) * h) // grid_size
        for gx in range(grid_size):
            x0 = (gx * w) // grid_size
            x1 = ((gx + 1) * w) // grid_size
            cell = lab[y0:y1, x0:x1]
            if piece_mask is not None:
                m = piece_mask[y0:y1, x0:x1] > 0
                if m.any():
                    out[gy, gx] = cell[m].mean(axis=0)
            else:
                out[gy, gx] = cell.reshape(-1, 3).mean(axis=0)
    return out


def _color_distance_map(master_bgr: np.ndarray, live_bgr: np.ndarray,
                        grid_size: int, piece_mask: Optional[np.ndarray]) -> np.ndarray:
    """Per-pixel LAB Euclidean distance between master and live, upscaled
    from the DINOv2-aligned patch grid. High values = color mismatch.

    LAB is used because Euclidean distance there approximates perceptual
    color difference (delta-E). Mean LAB per patch is robust to the small
    sub-pixel alignment noise that bites raw-pixel comparisons.
    """
    master_lab = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    live_lab = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    master_colors = _patch_mean_lab(master_lab, grid_size, piece_mask)
    live_colors = _patch_mean_lab(live_lab, grid_size, piece_mask)
    color_dist_grid = np.linalg.norm(master_colors - live_colors, axis=2)
    h, w = master_bgr.shape[:2]
    return cv2.resize(color_dist_grid, (w, h), interpolation=cv2.INTER_LINEAR)


def decoration_check(master_bgr: np.ndarray, live_bgr: np.ndarray,
                     master_features: Optional[DinoFeatures] = None,
                     piece_mask: Optional[np.ndarray] = None) -> DecorationCheckResult:
    if _h.model is None:
        init_model()

    if master_features is None:
        master_features = extract_features(master_bgr)
    live_features = extract_features(live_bgr)

    a = master_features.patch_tokens
    b = live_features.patch_tokens
    sims = np.sum(a * b, axis=1)                            # (P*P,)
    grid = master_features.grid_size
    sim_grid = sims.reshape(grid, grid)

    h, w = master_bgr.shape[:2]
    sim_map = cv2.resize(sim_grid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # Heatmap normalised to [0,1] for display; 1 = perfect match.
    heatmap = np.clip(sim_map, -1.0, 1.0)
    heatmap = (heatmap + 1.0) / 2.0

    # Restrict everything to the piece silhouette so empty background patches
    # can never produce decoration "defects". If no mask was supplied, fall back
    # to the previous full-image behavior.
    if piece_mask is not None:
        binary = (piece_mask > 0).astype(np.uint8)
        # Shrink slightly so silhouette-edge patches (which always look low-sim
        # due to anti-aliasing) don't dominate.
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.erode(binary, erode_k)
        roi_pixels = binary > 0
        if roi_pixels.any():
            roi_vals = sim_map[roi_pixels]
            global_sim = float(roi_vals.mean())
            problem_ratio = float(np.mean(roi_vals < PATCH_SIM_THRESHOLD))
        else:
            global_sim = float(sims.mean())
            problem_ratio = float(np.mean(sims < PATCH_SIM_THRESHOLD))
        low_mask = ((sim_map < PATCH_SIM_THRESHOLD) & roi_pixels).astype(np.uint8) * 255
        # For the heatmap render, neutralize background so it doesn't draw the eye.
        heatmap = np.where(roi_pixels, heatmap, 1.0).astype(np.float32)
        diff_piece_mask = binary
    else:
        global_sim = float(sims.mean())
        problem_ratio = float(np.mean(sims < PATCH_SIM_THRESHOLD))
        low_mask = (sim_map < PATCH_SIM_THRESHOLD).astype(np.uint8) * 255
        diff_piece_mask = None

    boxes = _box_components(low_mask, sim_map)
    max_box_area = max((b.w * b.h) for b in boxes) if boxes else 0

    # ---- Color deviation (LAB delta-E per patch) ----
    # Runs on the same patch grid as DINOv2 so the two signals are spatially
    # aligned. Catches a swapped-stone defect (e.g., ruby where a diamond
    # should be) that DINOv2 alone may underweight because its features are
    # more semantic than chromatic.
    color_dist_map = _color_distance_map(
        master_bgr, live_bgr, grid, piece_mask if piece_mask is not None else None
    )
    # Clamp to piece interior so background mismatch (different lighting on
    # the white backdrop) never registers as a defect.
    if piece_mask is not None:
        color_dist_map = np.where(diff_piece_mask > 0, color_dist_map, 0.0).astype(np.float32)

    color_low_mask = (color_dist_map > COLOR_DIST_HARD).astype(np.uint8) * 255
    color_boxes = _color_box_components(color_low_mask, color_dist_map)
    max_color_distance = float(color_dist_map.max())

    # All defect boxes — DINOv2 structural (orange) + color (magenta) — in
    # one list. The CheckResults card draws them all on the heatmap.
    boxes = boxes + color_boxes

    # ---- Combined deviation for the heatmap ----
    # Each source normalized to [0, 1] then maxed: a region lights up red
    # whether the structural patch features differ OR the color shifts.
    dino_deviation = np.clip(1.0 - sim_map, 0.0, 1.0)
    color_deviation = np.clip(color_dist_map / MAX_USEFUL_COLOR_DIST, 0.0, 1.0)
    combined_deviation = np.maximum(dino_deviation, color_deviation)

    diff_overlay = visualizer.build_decoration_deviation_overlay(
        live_bgr, combined_deviation, piece_mask=diff_piece_mask
    )

    if (
        global_sim < GLOBAL_HARD
        or problem_ratio > RATIO_HARD
        or max_box_area >= LOCAL_DEFECT_HARD_PX
        or max_color_distance > COLOR_DIST_HARD
    ):
        verdict = "FAIL"
    elif (
        global_sim < GLOBAL_BORDER
        or problem_ratio > RATIO_BORDER
        or max_box_area >= LOCAL_DEFECT_BORDER_PX
        or max_color_distance > COLOR_DIST_BORDER
    ):
        verdict = "BORDERLINE"
    else:
        verdict = "PASS"

    return DecorationCheckResult(
        global_similarity=global_sim,
        problem_patch_ratio=problem_ratio,
        max_color_distance=max_color_distance,
        heatmap=heatmap,
        diff_overlay=diff_overlay,
        verdict=verdict,
        boxes=boxes,
    )
