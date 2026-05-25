"""Composite output rendering.

Builds the difference-overlay panel (50/50 blend of master + aligned live
with all bounding boxes drawn) plus per-check heatmap PNGs. All outputs
are returned as numpy ndarrays so the API layer can encode them however
it wants.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .types import BoundingBox

# BGR colors matching the convention in the plan.
BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_YELLOW = (0, 255, 255)
BGR_ORANGE = (0, 165, 255)
BGR_MAGENTA = (255, 0, 255)
BGR_CYAN = (255, 255, 0)

COLOR_MAP = {
    "red": BGR_RED,
    "green": BGR_GREEN,
    "yellow": BGR_YELLOW,
    "orange": BGR_ORANGE,
    "magenta": BGR_MAGENTA,
    "cyan": BGR_CYAN,
}


def draw_boxes(img: np.ndarray, boxes: Iterable[BoundingBox]) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        color = COLOR_MAP.get(b.color, BGR_RED)
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        label = b.label
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
            ty = max(0, b.y - 6)
            cv2.rectangle(out, (b.x, ty - th - 3), (b.x + tw + 6, ty + 2), color, -1)
            cv2.putText(out, label, (b.x + 3, ty - 1), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_contour_outlines(
    img: np.ndarray,
    master_mask: np.ndarray,
    live_mask: np.ndarray,
) -> np.ndarray:
    out = img.copy()
    contours_m, _ = cv2.findContours(master_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_l, _ = cv2.findContours(live_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours_m, -1, BGR_GREEN, 1)
    cv2.drawContours(out, contours_l, -1, BGR_YELLOW, 1)
    return out


def build_difference_overlay(
    master_bgr: np.ndarray,
    live_aligned_bgr: np.ndarray,
    all_boxes: Iterable[BoundingBox],
) -> np.ndarray:
    blend = cv2.addWeighted(master_bgr, 0.5, live_aligned_bgr, 0.5, 0.0)
    return draw_boxes(blend, all_boxes)


def colorize_heatmap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    h = np.clip(heatmap, 0.0, 1.0)
    h8 = (h * 255).astype(np.uint8)
    return cv2.applyColorMap(h8, colormap)


def colorize_decoration_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Red (<0.85) -> yellow (0.85-0.95) -> green (>0.95) per plan."""
    h = np.clip(heatmap, 0.0, 1.0)
    out = np.zeros((*h.shape, 3), dtype=np.uint8)
    # Red channel: high when low sim, low when high sim. Build piecewise.
    # Map sim in [0.85, 0.95] linearly through yellow.
    low = h < 0.85
    mid = (h >= 0.85) & (h < 0.95)
    hi = h >= 0.95
    # OpenCV is BGR.
    out[..., 2] = np.where(low, 255, np.where(mid, 255, 0))                       # R
    out[..., 1] = np.where(low, 0, np.where(mid, ((h - 0.85) / 0.10 * 255).astype(np.uint8), 255))  # G
    out[..., 0] = 0                                                                # B
    return out


def colorize_surface_heatmap(defect_map: np.ndarray) -> np.ndarray:
    """Blue (low anomaly) -> red (high anomaly)."""
    d = np.clip(defect_map, 0.0, 1.0)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
