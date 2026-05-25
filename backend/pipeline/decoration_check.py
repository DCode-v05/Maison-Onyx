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
    else:
        global_sim = float(sims.mean())
        problem_ratio = float(np.mean(sims < PATCH_SIM_THRESHOLD))
        low_mask = (sim_map < PATCH_SIM_THRESHOLD).astype(np.uint8) * 255

    boxes = _box_components(low_mask, sim_map)
    max_box_area = max((b.w * b.h) for b in boxes) if boxes else 0

    if (
        global_sim < GLOBAL_HARD
        or problem_ratio > RATIO_HARD
        or max_box_area >= LOCAL_DEFECT_HARD_PX
    ):
        verdict = "FAIL"
    elif (
        global_sim < GLOBAL_BORDER
        or problem_ratio > RATIO_BORDER
        or max_box_area >= LOCAL_DEFECT_BORDER_PX
    ):
        verdict = "BORDERLINE"
    else:
        verdict = "PASS"

    return DecorationCheckResult(
        global_similarity=global_sim,
        problem_patch_ratio=problem_ratio,
        heatmap=heatmap,
        verdict=verdict,
        boxes=boxes,
    )
