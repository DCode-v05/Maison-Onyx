"""Stage 7 — Surface quality check.

Local SSIM map + high-frequency Laplacian difference. Combined into a
single defect map, thresholded into binary defect regions. Regions become
bounding boxes (axis-aligned, or rotated for high aspect-ratio scratches).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .types import BoundingBox, SurfaceCheckResult

MIN_DEFECT_PX = 120
MAJOR_DEFECT_PX = 400

SSIM_WIN = 11
SCRATCH_SENSITIVITY = 0.65
EDGE_ERODE = 17

# Local-variance threshold for "this region is textured (pave, etched, faceted)
# — surface anomaly detection is unreliable here." Computed from the master
# image only, so production behavior is deterministic per SKU.
TEXTURE_WIN = 15
TEXTURE_THRESHOLD = 300.0    # std-dev-squared of grayscale in a TEXTURE_WIN window
TEXTURE_DILATE = 7           # grow the texture mask slightly so we exclude
                              # texture borders too (where specular spills over)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.erode(mask, k)


def _texture_mask(gray: np.ndarray) -> np.ndarray:
    """1 where the master surface is smooth (surface check valid), 0 where
    it is textured (pave, faceted stones, polished cutouts producing strong
    specular highlights). Texture is measured as local variance of grayscale.
    """
    g = gray.astype(np.float32)
    win = TEXTURE_WIN
    mean = cv2.blur(g, (win, win))
    mean_sq = cv2.blur(g * g, (win, win))
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    smooth = (var < TEXTURE_THRESHOLD).astype(np.uint8) * 255
    # Erode the smooth mask (= dilate the textured exclusion) so we also
    # suppress the immediate halo around bright stones / faceted edges.
    if TEXTURE_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TEXTURE_DILATE, TEXTURE_DILATE))
        smooth = cv2.erode(smooth, k)
    return smooth


def _high_freq_diff(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    l1 = cv2.Laplacian(g1.astype(np.float32), cv2.CV_32F, ksize=3)
    l2 = cv2.Laplacian(g2.astype(np.float32), cv2.CV_32F, ksize=3)
    d = np.abs(l1 - l2)
    if d.max() > 0:
        d = d / d.max()
    return d.astype(np.float32)


def _boxes_from_components(mask: np.ndarray, gray: np.ndarray) -> List[BoundingBox]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: List[BoundingBox] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < MIN_DEFECT_PX:
            continue
        severity = "MAJOR" if area >= MAJOR_DEFECT_PX else "MINOR"
        label = f"{severity} {int(area)}px"
        boxes.append(BoundingBox(int(x), int(y), int(w), int(h),
                                 color="magenta", label=label, score=float(area)))
    return boxes


def surface_check(master_bgr: np.ndarray, live_bgr: np.ndarray,
                  shared_mask: np.ndarray) -> SurfaceCheckResult:
    gm = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    gl = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY)

    eroded = _erode((shared_mask > 0).astype(np.uint8) * 255, EDGE_ERODE)
    # Restrict the analysis to smooth regions of the master. Textured regions
    # (pave stones, faceted cutouts) produce false positives because their
    # specular highlights shift between exposures even without any defect.
    smooth = _texture_mask(gm)
    eroded = cv2.bitwise_and(eroded, smooth)

    # SSIM map (lower => more anomaly), invert to anomaly intensity.
    try:
        score, ssim_map = ssim(gm, gl, win_size=SSIM_WIN, full=True, data_range=255)
    except Exception:
        ssim_map = np.ones_like(gm, dtype=np.float32)
    anomaly_ssim = np.clip(1.0 - ssim_map.astype(np.float32), 0.0, 1.0)

    anomaly_hf = _high_freq_diff(gm, gl)

    defect_map = 0.6 * anomaly_ssim + 0.4 * anomaly_hf
    # Zero out the area outside the eroded piece mask to suppress edge noise.
    defect_map[eroded == 0] = 0.0
    if defect_map.max() > 0:
        defect_map = defect_map / max(defect_map.max(), 1e-6)

    # Fixed sensitivity floor. Otsu was too eager when the whole piece had
    # mild SSIM noise from sub-pixel misalignment — it would pick a low
    # threshold and flag the entire piece as a defect.
    bin_defects = (defect_map >= SCRATCH_SENSITIVITY).astype(np.uint8) * 255
    bin_defects[eroded == 0] = 0
    # Close small gaps so a single scratch becomes one component, not many.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bin_defects = cv2.morphologyEx(bin_defects, cv2.MORPH_CLOSE, close_k)

    boxes = _boxes_from_components(bin_defects, gm)

    total_pixels = int((eroded > 0).sum())
    defect_pixels = int((bin_defects > 0).sum())
    defect_ratio = float(defect_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
    max_defect = max((b.score or 0) for b in boxes) if boxes else 0

    has_major = any(b.label.startswith("MAJOR") for b in boxes)
    if defect_ratio > 0.012 or has_major:
        verdict = "FAIL"
    elif defect_ratio > 0.004 or len(boxes) >= 2:
        verdict = "BORDERLINE"
    else:
        verdict = "PASS"

    return SurfaceCheckResult(
        defect_ratio=defect_ratio,
        num_defect_regions=len(boxes),
        max_defect_size=int(max_defect),
        defect_map=defect_map.astype(np.float32),
        verdict=verdict,
        boxes=boxes,
    )
