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
BGR_GOLD = (97, 169, 201)   # mirrors the UI gold accent (#C9A961)

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
        if b.points:
            # Rotated/polygon box — used for scratches (high aspect ratio).
            # The polygon follows the scratch direction; (x, y, w, h) is the
            # axis-aligned bounding rect of the polygon (used to anchor the label).
            pts = np.array(b.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        else:
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


# ---- Edge detection (full structure) ----

EDGE_ROI_ERODE_PX = 3       # how far inside the silhouette to start trusting edges
EDGE_SIGMA = 0.33           # median-based adaptive Canny thresholds (well-known default)
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)


def edge_map(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Canny edges over the piece, captures full structure (outer outline +
    pave stones + cutouts + engravings + band engravings).

    Three things make this work on subtle gold-on-gold detail where naive
    Canny fails:

    1. CLAHE locally amplifies contrast so the band's cross-hatch engravings
       and the metal/stone interface produce real gradients.
    2. Thresholds are derived from the image median (sigma=0.33) instead of
       hard-coded — auto-tunes per piece, per lighting.
    3. cv2.Canny is run with L2gradient=True for a more accurate gradient
       magnitude (matters when contrast is low).

    The external silhouette is OR'd in at the end so the outer profile is
    guaranteed even if Canny suppressed it at the band/background boundary.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE before any blur — the blur happens next as part of edge prep.
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    gray = clahe.apply(gray)

    # Mild blur to suppress sensor/JPEG speckle without erasing real edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Median-based adaptive thresholds. The median is computed over the
    # piece interior only, so background lighting doesn't bias the result.
    roi_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_ROI_ERODE_PX, EDGE_ROI_ERODE_PX))
    roi = cv2.erode((mask > 0).astype(np.uint8) * 255, roi_k)
    piece_pixels = blurred[roi > 0]
    if piece_pixels.size > 0:
        med = float(np.median(piece_pixels))
    else:
        med = float(np.median(blurred))
    low = int(max(0, (1.0 - EDGE_SIGMA) * med))
    high = int(min(255, (1.0 + EDGE_SIGMA) * med))

    edges = cv2.Canny(blurred, low, high, apertureSize=3, L2gradient=True)

    # Restrict to the piece interior.
    edges = cv2.bitwise_and(edges, edges, mask=roi)

    # Guarantee the external silhouette is present even when Canny missed it.
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        outline = np.zeros_like(edges)
        cv2.drawContours(outline, contours, -1, 255, 1)
        edges = cv2.bitwise_or(edges, outline)
    return edges


def draw_edges(
    img: np.ndarray,
    edges: np.ndarray,
    color: Tuple[int, int, int] = BGR_GREEN,
    thickness: int = 1,
) -> np.ndarray:
    """Paint every edge pixel in `color`. `thickness` dilates the edge map."""
    out = img.copy()
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edges = cv2.dilate(edges, k)
    sel = edges > 0
    if sel.any():
        out[sel] = np.array(color, dtype=np.uint8)
    return out


def build_profile_deviation_overlay(
    master_bgr: np.ndarray,
    missing_edges: np.ndarray,
    excess_edges: np.ndarray,
    dim: float = 0.40,
    missing_thickness: int = 2,
    excess_thickness: int = 1,
) -> np.ndarray:
    """Render the per-check Profile cell: master image dimmed, with master-only
    edges (i.e., edges present in the master but absent in the live image)
    painted bright red, and live-only edges in orange.

    The red pixels are the answer to "where does the live piece's structure
    deviate from the master?" Missing is rendered with priority (thicker,
    drawn on top of excess) because that's the question the operator is
    actually asking when a profile FAIL fires.
    """
    out = (master_bgr.astype(np.float32) * dim).astype(np.uint8)

    if excess_thickness > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (excess_thickness, excess_thickness)
        )
        ee = cv2.dilate(excess_edges, k)
    else:
        ee = excess_edges
    sel = ee > 0
    if sel.any():
        out[sel] = np.array(BGR_ORANGE, dtype=np.uint8)

    if missing_thickness > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (missing_thickness, missing_thickness)
        )
        me = cv2.dilate(missing_edges, k)
    else:
        me = missing_edges
    sel = me > 0
    if sel.any():
        out[sel] = np.array(BGR_RED, dtype=np.uint8)

    return out


def _filled_silhouette(mask: np.ndarray) -> np.ndarray:
    """Fill the outer contour so the silhouette has no internal holes.

    The segmentation mask is Swiss-cheese on jewelry photographs because
    Otsu thresholds out the bright pave stones — they get classified as
    background and end up as holes in the foreground mask. Filling the
    outer contour gives a clean solid shape that covers the whole piece.
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return out
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def build_decoration_deviation_overlay(
    base_bgr: np.ndarray,
    deviation_map: np.ndarray,
    piece_mask: np.ndarray | None = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.65,
    background_dim: float = 0.35,
) -> np.ndarray:
    """JET heatmap blended over `base_bgr`, bounded by the *filled* silhouette.

    `deviation_map` is already in [0, 1] where 0 = perfect match and 1 = max
    deviation. The caller is responsible for combining whatever deviation
    signals it wants (DINOv2 1-sim, LAB color distance, surface anomalies)
    into this single field — keeps the visualizer agnostic to the source.

    `base_bgr` is whatever image the operator should see *under* the heatmap.
    For decoration we pass the rotation-aligned live so the hotspot lands on
    the visible defect, not at a corresponding-but-translated master location.

    Inside the silhouette: base blended with the JET heatmap at `alpha`.
    Outside the silhouette: base dimmed to `background_dim` brightness.
    """
    deviation = np.clip(deviation_map, 0.0, 1.0)
    dev_u8 = (deviation * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(dev_u8, colormap)

    if piece_mask is None:
        return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0.0)

    silhouette = _filled_silhouette(piece_mask)
    sel = silhouette > 0

    out = (base_bgr.astype(np.float32) * background_dim).astype(np.uint8)
    if sel.any():
        blended = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0.0)
        out[sel] = blended[sel]
    return out


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = BGR_GOLD,
    alpha: float = 0.40,
    dim_outside: float = 0.55,
) -> np.ndarray:
    """Tint pixels under the mask, dim pixels outside it.

    Used for the "segmented" panel in the live stage strip — gives the
    operator an at-a-glance read of what stage 2 considered the piece.
    """
    out = img.astype(np.float32)
    sel = mask > 0
    if sel.any():
        tint = np.array(color, dtype=np.float32)
        out[sel] = out[sel] * (1.0 - alpha) + tint * alpha
    if dim_outside < 1.0:
        out[~sel] = out[~sel] * dim_outside
    return np.clip(out, 0, 255).astype(np.uint8)


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
