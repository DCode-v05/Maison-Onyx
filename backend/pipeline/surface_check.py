"""Stage 7 — Surface quality check.

Combines four signals into a defect map:

  - SSIM           — local structural anomaly on grayscale (catches scratches,
                     pits, dents, broad polishing-mark differences)
  - Laplacian diff — high-frequency anomaly (catches sharp scratches, fine pits)
  - LAB color      — per-pixel CIELAB delta-E (catches tarnish, fingerprint
                     residue, polish residue, plating color shifts — defects
                     the grayscale signals miss)
  - Stone-zone     — separate color-only check inside textured regions
                     (pavé, faceted cutouts), tuned with a permissive
                     threshold so specular variation doesn't false-trigger
                     but a wrong-color or missing stone still fires

Connected components in the binary defect mask become bounding boxes —
axis-aligned for blobby defects, *rotated* (cv2.minAreaRect) for high
aspect-ratio scratches so the operator gets a clearer visual indicator.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim

from .types import BoundingBox, SurfaceCheckResult

# Defect size thresholds.
MIN_DEFECT_PX = 50              # was 120 — lower so lint / small pits / fine scratches register
MAJOR_DEFECT_PX = 400
SCRATCH_MIN_LENGTH = 25         # px — short side ignored, long side must beat this
SCRATCH_ASPECT_RATIO = 3.0      # length/width threshold to classify as scratch

SSIM_WIN = 11
SCRATCH_SENSITIVITY = 0.65      # smooth-zone defect threshold on combined map
EDGE_ERODE = 17

# Local-variance threshold for "this region is textured (pave, etched, faceted)
# — anomaly detection on grayscale is unreliable here." Computed from the
# master image only, so production behavior is deterministic per SKU.
TEXTURE_WIN = 15
TEXTURE_THRESHOLD = 300.0
TEXTURE_DILATE = 7

# LAB color normalization. Per-pixel Euclidean LAB distance normalized to
# [0, 1] for the combined defect map. 25 ≈ "clearly different color" delta-E.
COLOR_NORM_SMOOTH = 25.0        # smooth-zone color normalization
COLOR_NORM_STONE = 40.0         # stone-zone color normalization (more tolerant
                                # of specular variation, fires only on a real
                                # wrong/missing stone)
STONE_DEFECT_THRESHOLD = 0.6    # stone-zone binary threshold on color anomaly

# Verdict thresholds for the new max_color_distance metric.
COLOR_DIST_HARD = 30.0          # FAIL — clearly wrong color somewhere
COLOR_DIST_BORDER = 15.0        # BORDERLINE — noticeable color shift

# Defensive pre-processing — protects against image quality mismatches
# (e.g., PNG master vs JPEG live) and global exposure/lighting drift.
PREBLUR_KSIZE = 5               # Gaussian to smooth JPEG block artifacts + minor
PREBLUR_SIGMA = 1.2             # specular shifts before SSIM/Laplacian

# Adaptive threshold: median + ADAPTIVE_K * MAD inside the smooth zone.
# Lets the binary cut-off track the per-image noise floor — clean images
# get a strict threshold, noisy ones get a looser one (but bounded).
ADAPTIVE_K = 6.0                # was 5 — slightly more aggressive noise rejection
ADAPTIVE_FLOOR = 0.45           # never below this even if image is unusually clean
ADAPTIVE_CEILING = 0.85         # never above this even if image is unusually noisy


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.erode(mask, k)


def _texture_mask(gray: np.ndarray) -> np.ndarray:
    """1 where the master surface is smooth, 0 where it is textured (pavé,
    faceted stones, polished cutouts producing strong specular highlights).
    Texture is measured as local variance of grayscale.
    """
    g = gray.astype(np.float32)
    win = TEXTURE_WIN
    mean = cv2.blur(g, (win, win))
    mean_sq = cv2.blur(g * g, (win, win))
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    smooth = (var < TEXTURE_THRESHOLD).astype(np.uint8) * 255
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


def _classify_component(
    contour: np.ndarray,
    area: int,
    smooth_zone: np.ndarray,
    stone_zone: np.ndarray,
    mask_shape: Tuple[int, int],
) -> BoundingBox:
    """Build a BoundingBox for a single connected-component contour. Picks
    axis-aligned or rotated rect depending on aspect ratio, picks color and
    label depending on which zone (smooth vs stone) the defect lives in.
    """
    # Determine zone by majority overlap.
    comp_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(comp_mask, [contour], -1, 255, thickness=cv2.FILLED)
    overlap_smooth = int(cv2.countNonZero(cv2.bitwise_and(comp_mask, smooth_zone)))
    overlap_stone = int(cv2.countNonZero(cv2.bitwise_and(comp_mask, stone_zone)))
    is_stone = overlap_stone > overlap_smooth

    # Rotated rect for aspect-ratio check (scratch detection).
    rect = cv2.minAreaRect(contour)
    (_cx, _cy), (rw, rh), _angle = rect
    long_side = max(rw, rh)
    short_side = max(min(rw, rh), 1.0)
    aspect = long_side / short_side

    is_scratch = (
        not is_stone
        and aspect >= SCRATCH_ASPECT_RATIO
        and long_side >= SCRATCH_MIN_LENGTH
    )

    if is_scratch:
        corners = cv2.boxPoints(rect).astype(np.int32)
        xs = corners[:, 0]
        ys = corners[:, 1]
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - bx), int(ys.max() - by)
        return BoundingBox(
            x=bx, y=by, w=bw, h=bh,
            color="magenta",
            label=f"SCRATCH {int(long_side)}px",
            score=float(area),
            points=[(int(p[0]), int(p[1])) for p in corners],
        )

    # Axis-aligned for blobby defects.
    x, y, w, h = cv2.boundingRect(contour)
    severity = "MAJOR" if area >= MAJOR_DEFECT_PX else "MINOR"
    if is_stone:
        # Distinguish stone defects visually + in label.
        return BoundingBox(
            x=int(x), y=int(y), w=int(w), h=int(h),
            color="cyan",
            label=f"STONE {int(area)}px",
            score=float(area),
        )
    return BoundingBox(
        x=int(x), y=int(y), w=int(w), h=int(h),
        color="magenta",
        label=f"{severity} {int(area)}px",
        score=float(area),
    )


def _boxes_from_components(
    mask: np.ndarray,
    smooth_zone: np.ndarray,
    stone_zone: np.ndarray,
) -> List[BoundingBox]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes: List[BoundingBox] = []
    h, w = mask.shape[:2]
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < MIN_DEFECT_PX:
            continue
        boxes.append(
            _classify_component(contour, area, smooth_zone, stone_zone, (h, w))
        )
    return boxes


def _normalize_live_to_master(live_bgr: np.ndarray, master_bgr: np.ndarray) -> np.ndarray:
    """Warp live's color/intensity distribution to match master's via per-channel
    histogram matching. Removes global exposure / white-balance / JPEG-quantization
    differences that would otherwise register as widespread surface anomaly.

    Returns a uint8 BGR image with the same content but distribution-matched
    to master. Falls back to the original on any matching error.
    """
    try:
        matched = match_histograms(live_bgr, master_bgr, channel_axis=-1)
        return np.clip(matched, 0, 255).astype(np.uint8)
    except Exception:
        return live_bgr


def _preblur(gray: np.ndarray) -> np.ndarray:
    """Small Gaussian to suppress JPEG 8x8 block boundaries and sensor speckle
    without erasing real defects — a true scratch is much sharper than the
    DCT-ringing noise this filter targets."""
    return cv2.GaussianBlur(gray, (PREBLUR_KSIZE, PREBLUR_KSIZE), PREBLUR_SIGMA)


def _adaptive_threshold(defect_map: np.ndarray, mask: np.ndarray) -> float:
    """Robust threshold = median + ADAPTIVE_K * MAD inside `mask`. Clamped
    to [ADAPTIVE_FLOOR, ADAPTIVE_CEILING]. MAD (median absolute deviation)
    is used instead of std because it's not pulled around by real defect
    pixels — those are outliers, and we want the threshold to track the
    noise distribution, not get inflated by what we're trying to detect.
    """
    vals = defect_map[mask > 0]
    if vals.size == 0:
        return SCRATCH_SENSITIVITY
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    thresh = med + ADAPTIVE_K * mad
    return max(ADAPTIVE_FLOOR, min(ADAPTIVE_CEILING, thresh))


def surface_check(master_bgr: np.ndarray, live_bgr: np.ndarray,
                  shared_mask: np.ndarray) -> SurfaceCheckResult:
    # ---- Defensive pre-processing (Option B) ----
    # Histogram-match live to master so global exposure/white-balance/JPEG
    # quantization drift doesn't register as widespread surface anomaly.
    live_norm = _normalize_live_to_master(live_bgr, master_bgr)

    gm_raw = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    gl_raw = cv2.cvtColor(live_norm, cv2.COLOR_BGR2GRAY)
    # Light blur before SSIM/Laplacian — smooths JPEG 8x8 block boundaries
    # without erasing real defects (which are much sharper than DCT ringing).
    gm = _preblur(gm_raw)
    gl = _preblur(gl_raw)

    # Eroded piece mask: pull the analysis off the silhouette boundary so
    # 1-px registration offsets don't create edge-noise defects.
    piece = _erode((shared_mask > 0).astype(np.uint8) * 255, EDGE_ERODE)
    smooth = _texture_mask(gm_raw)                          # 255 = smooth (texture mask from raw master)
    smooth_zone = cv2.bitwise_and(piece, smooth)            # smooth surface, inside piece
    stone_zone = cv2.bitwise_and(piece, cv2.bitwise_not(smooth))  # textured (pavé/facet) inside piece

    # ---- Grayscale anomaly signals (smooth surface only) ----
    try:
        _score, ssim_map = ssim(gm, gl, win_size=SSIM_WIN, full=True, data_range=255)
    except Exception:
        ssim_map = np.ones_like(gm, dtype=np.float32)
    anomaly_ssim = np.clip(1.0 - ssim_map.astype(np.float32), 0.0, 1.0)
    anomaly_hf = _high_freq_diff(gm, gl)

    # ---- LAB color anomaly (per-pixel delta-E) ----
    # Use the histogram-matched live so a global color cast (different
    # white balance, JPEG chroma subsampling) doesn't dominate delta-E.
    master_lab = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    live_lab = cv2.cvtColor(live_norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_dist = np.linalg.norm(master_lab - live_lab, axis=2)
    # Normalized [0, 1] anomaly fields, one per zone (different tolerances).
    color_anomaly_smooth = np.clip(color_dist / COLOR_NORM_SMOOTH, 0.0, 1.0)
    color_anomaly_stone = np.clip(color_dist / COLOR_NORM_STONE, 0.0, 1.0)

    # ---- Smooth-zone combined defect signal ----
    # Max of (SSIM+Laplacian structural blend) and color — either type of
    # anomaly is enough to flag. Catches scratches/pits (structural) AND
    # tarnish/fingerprints/polish residue (color).
    structural = 0.6 * anomaly_ssim + 0.4 * anomaly_hf
    defect_smooth_map = np.maximum(structural, color_anomaly_smooth)
    defect_smooth_map[smooth_zone == 0] = 0.0

    # ---- Stone-zone defect signal: color only, tighter threshold ----
    # SSIM/Laplacian misfire here from specular shift; color is robust enough
    # at the COLOR_NORM_STONE tolerance to fire only on a real wrong/missing stone.
    defect_stone_map = color_anomaly_stone.copy()
    defect_stone_map[stone_zone == 0] = 0.0

    # ---- Binary defect mask per zone ----
    # Smooth zone gets an *adaptive* threshold so a noisy master/live pair
    # (e.g., JPEG-vs-PNG quality mismatch) gets a higher cutoff, suppressing
    # widespread false positives, while a clean pair keeps a low cutoff,
    # preserving sensitivity to subtle defects.
    smooth_threshold = _adaptive_threshold(defect_smooth_map, smooth_zone)
    bin_smooth = (defect_smooth_map >= smooth_threshold).astype(np.uint8) * 255
    # Stone zone keeps the fixed color threshold — color-only signal there
    # is intrinsically stable and we don't want to risk masking a real
    # wrong-stone defect by adapting away from it.
    bin_stone = (defect_stone_map >= STONE_DEFECT_THRESHOLD).astype(np.uint8) * 255
    bin_defects = cv2.bitwise_or(bin_smooth, bin_stone)

    # Close small gaps so a single scratch becomes one component, not many.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bin_defects = cv2.morphologyEx(bin_defects, cv2.MORPH_CLOSE, close_k)

    boxes = _boxes_from_components(bin_defects, smooth_zone, stone_zone)

    # ---- Display map for the per-check heatmap ----
    # Combine both zones with light de-emphasis on stone zone so structural
    # defects on the band don't get visually drowned out by stone-zone signal.
    display_raw = np.maximum(defect_smooth_map, defect_stone_map * 0.85)
    display_raw[piece == 0] = 0.0

    # Show "anomaly above noise floor" rather than raw anomaly. Otherwise,
    # the JET visualization lights up the entire piece in cyan/blue when
    # there's any image-quality noise (JPEG artifacts, perspective drift,
    # specular shifts) — even when no real defect actually crosses the
    # adaptive binary threshold. Subtracting the median signal level inside
    # the piece pushes clean areas to deep blue and only leaves the
    # genuinely-above-noise hotspots visible.
    piece_vals = display_raw[piece > 0]
    noise_floor = float(np.median(piece_vals)) if piece_vals.size > 0 else 0.0
    display_map = np.clip(display_raw - noise_floor, 0.0, 1.0)
    if display_map.max() > 0:
        display_map = display_map / display_map.max()

    # ---- Metrics ----
    total_pixels = int((piece > 0).sum())
    defect_pixels = int((bin_defects > 0).sum())
    defect_ratio = float(defect_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
    max_defect = int(max((b.score or 0) for b in boxes)) if boxes else 0
    max_color_distance = float(color_dist[piece > 0].max()) if total_pixels > 0 else 0.0
    num_scratches = sum(1 for b in boxes if b.label.startswith("SCRATCH"))
    num_stone_defects = sum(1 for b in boxes if b.label.startswith("STONE"))

    # ---- Verdict ----
    has_major = any(
        b.label.startswith("MAJOR") or b.label.startswith("SCRATCH") or b.label.startswith("STONE")
        for b in boxes
    )
    if (
        defect_ratio > 0.012
        or has_major
        or max_color_distance > COLOR_DIST_HARD
    ):
        verdict = "FAIL"
    elif (
        defect_ratio > 0.004
        or len(boxes) >= 2
        or max_color_distance > COLOR_DIST_BORDER
    ):
        verdict = "BORDERLINE"
    else:
        verdict = "PASS"

    return SurfaceCheckResult(
        defect_ratio=defect_ratio,
        num_defect_regions=len(boxes),
        max_defect_size=max_defect,
        max_color_distance=max_color_distance,
        num_scratches=num_scratches,
        num_stone_defects=num_stone_defects,
        defect_map=display_map.astype(np.float32),
        verdict=verdict,
        boxes=boxes,
    )
