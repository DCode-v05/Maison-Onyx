# Jewelry Visual Inspection — Validation Plan

## Objective

Validate the proposed reference-based inspection pipeline for **accuracy** (does it catch defects?) and **speed** (does it run within 1 second?) using manually uploaded master and live image pairs.

This plan covers building a minimal working pipeline, measuring per-stage latency, and evaluating detection accuracy across all three checks — before any production integration.

---

## 1. Test Dataset Preparation

### 1.1 What to Collect

For each jewelry SKU, prepare one master image (the golden reference) and multiple live images covering the following categories.

**Good pieces (true pass):** 15–20 images per SKU of pieces that should pass inspection. Vary the placement orientation deliberately — rotate the piece by different amounts between captures to simulate real operator placement. These images validate that the system doesn't false-reject good pieces.

**Defective pieces (true fail):** 5–10 images per defect type per SKU. Separate them by which check they should trigger.

| Check | Defect Types to Include |
|-------|------------------------|
| Profile | Wrong size, bent prong, missing structural element, asymmetry |
| Decoration | Missing stone, chipped stone, wrong stone color, enamel bubble, enamel color mismatch, misaligned engraving |
| Surface | Scratch, pit, tarnish spot, polishing mark, surface contamination |

**Borderline cases:** 3–5 images per SKU of pieces that are marginal — slight scratches, minor enamel variation, barely acceptable profile. These test whether borderline pieces go to REVIEW rather than being auto-accepted.

### 1.2 Image Capture Requirements

All images must be captured under the same lighting and camera setup. Background must be consistent matte black or matte white across all images. Resolution should match your target production resolution (recommend 4K / 12MP minimum). Save as lossless PNG, not JPEG — compression artifacts will mask real defects and corrupt benchmarks.

### 1.3 Naming Convention

```
dataset/
├── SKU_001/
│   ├── master.png
│   ├── good/
│   │   ├── good_001_rot0.png
│   │   ├── good_002_rot45.png
│   │   ├── good_003_rot90.png
│   │   └── ...
│   ├── defect_profile/
│   │   ├── bent_prong_001.png
│   │   └── ...
│   ├── defect_decoration/
│   │   ├── missing_stone_001.png
│   │   └── ...
│   ├── defect_surface/
│   │   ├── scratch_001.png
│   │   └── ...
│   └── borderline/
│       ├── minor_scratch_001.png
│       └── ...
├── SKU_002/
│   └── ...
└── ground_truth.csv
```

### 1.4 Ground Truth File

A CSV with one row per live image recording the expected outcome.

```
image_path, sku, expected_decision, expected_profile, expected_decoration, expected_surface, defect_description
SKU_001/good/good_001_rot0.png, SKU_001, ACCEPT, PASS, PASS, PASS, none
SKU_001/defect_surface/scratch_001.png, SKU_001, REJECT, PASS, PASS, FAIL, deep_scratch_on_band
SKU_001/borderline/minor_scratch_001.png, SKU_001, REVIEW, PASS, PASS, BORDERLINE, faint_scratch_near_setting
```

### 1.5 Minimum Dataset Size

For a meaningful initial validation, aim for at least 3 SKUs with at least 30 images each (20 good + 10 defective). Total minimum: ~100 images. This is enough to identify gross problems in the pipeline. It is not enough for production-grade threshold calibration — that requires 500+ images per SKU.

---

## 2. Pipeline Stages to Implement and Measure

Each stage is implemented as an independent, timed module. Every stage logs its own wall-clock execution time.

### Stage 1: Image Loading and Preprocessing

**What it does:** Load the live image from disk, decode it, resize to working resolution.

**Working resolution decision:** Run all downstream stages at 1024×1024. This balances detail preservation with speed. The original full-resolution image is kept for surface quality check where pixel-level detail matters — use a cropped ROI at full resolution rather than a global downscale.

**Expected latency:** 5–15ms depending on image size and storage medium (SSD vs HDD).

**What to measure:** Decode time for PNG at your target resolution. If decode time exceeds 30ms, consider capturing in BMP (no decode overhead) or using a hardware-accelerated decoder.

---

### Stage 2: Segmentation and Centering

**What it does:** Separate the jewelry piece from the background. Produce a binary mask, bounding box, and centroid.

**Approach:** Adaptive thresholding on grayscale → morphological cleanup → largest connected component extraction. On a controlled black background, this is deterministic and fast.

**Expected latency:** 5–10ms on CPU at 1024×1024.

**Accuracy validation:** For every image in the test set, visually verify the segmentation mask. Check that the mask tightly covers the piece without cutting off edges (prongs, chain links) or including background artifacts. Record the IoU between automated mask and a manually drawn mask for 20 representative images — target IoU > 0.95.

**Failure modes to test:** Highly reflective pieces that reflect the dark background (causing holes in the mask). Pieces with very thin elements (chain links, wire settings) that get lost in morphological cleanup. Pieces placed near the edge of the frame.

---

### Stage 3: Rotation Estimation

**What it does:** Determine how much the live piece is rotated relative to the master, and correct it.

**Approach:** Compute the orientation angle of the segmented piece using image moments (the angle of the major axis from the moments ellipse). Compare against the precomputed master orientation angle. Rotate the live image by the difference. Handle 180° ambiguity by comparing NCC at both 0° and 180° correction and picking the better match.

**Expected latency:** 5–10ms on CPU.

**Accuracy validation:** This is the most critical stage to validate. For each test image, record the estimated rotation angle. Compare against the known rotation (if you rotated the piece deliberately, you know roughly by how much). The metric is **rotation error in degrees** — the difference between estimated and actual rotation.

Acceptance criteria:
- Mean rotation error < 2° across all test images
- Maximum rotation error < 5° on any single image
- For symmetric pieces, verify that the selected rotation candidate (among symmetry-equivalent options) produces consistent check results

**Test specifically:** Pieces at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° — cover the full range. Pieces with rotational symmetry. Pieces where the major axis is ambiguous (roughly circular outlines like round pendants).

---

### Stage 4: Fine Registration (SIFT)

**What it does:** Refine the coarse-aligned live image to subpixel alignment with the master using feature matching.

**Approach:** Run SIFT on both master (precomputed) and coarse-aligned live image at working resolution (1024×1024). Match descriptors, apply Lowe's ratio test (threshold 0.6), compute homography with RANSAC, warp the live image.

**Expected latency:** 50–80ms on CPU. SIFT keypoint detection is the bottleneck. Limit keypoints to 1500 max to control this.

**Accuracy validation:** For each test image, overlay the registered live image on the master at 50% opacity and visually inspect alignment quality. Quantitatively, measure the NCC (normalized cross-correlation) between registered live and master within the piece mask. Record the number of SIFT inlier matches and the inlier ratio.

Acceptance criteria:
- Registration NCC > 0.85 for good pieces
- Inlier ratio > 0.5 (RANSAC inliers / total matches)
- For good pieces, no visible misalignment at stone positions or profile edges

**Failure modes to test:** Pieces with very few texture features (plain polished bands). Pieces with repetitive patterns (eternity bands with identical stones). Pieces where the rotation estimation was slightly off (does fine registration recover?).

**Fallback behavior:** If SIFT produces fewer than 15 inlier matches, the registration is unreliable. Log this and route the piece to REVIEW. Measure what percentage of your test set hits this fallback — if it exceeds 5%, the imaging setup or rotation estimation needs improvement.

---

### Stage 5: Profile Check

**What it does:** Compare the silhouette of the registered live piece against the master silhouette.

**Approach:** Extract contours from both masks. Compute three metrics: shape distance (Hu moments via `cv2.matchShapes`), area deviation (fractional difference in contour area), and silhouette IoU (pixel overlap of the two binary masks). A piece passes if all three are within threshold.

**Expected latency:** 20–30ms on CPU.

**Accuracy validation:** Run on all test images and record all three metrics. For known-good pieces, establish the distribution of each metric — this tells you what "normal" variation looks like. For profile-defective pieces, verify that at least one metric clearly exceeds the good distribution.

Threshold calibration approach:
1. Plot the distribution of each metric for good pieces and for profile-defective pieces
2. Set the threshold at the point that achieves 0% false accept rate on the defective set
3. Record the resulting false reject rate on the good set — target < 5%
4. If you can't achieve both targets simultaneously, the check needs refinement or the imaging setup needs improvement

**Output to log per image:** shape_distance, area_deviation, silhouette_iou, pass/fail decision, diff_mask image (save to disk for visual review).

---

### Stage 6: Decoration Check

**What it does:** Compare decorative elements (stones, enamel, engravings) between registered live and master using learned feature matching.

**Approach:** Extract DINOv2 patch features from the registered live image (master features precomputed). Compute cosine similarity per patch location. Flag patches below similarity threshold as potential decoration mismatches. Compute global similarity score and problem patch ratio.

**Model selection:** Start with DINOv2 ViT-S/14 (small). If accuracy is insufficient, upgrade to ViT-B/14. Do not go larger — ViT-L and ViT-G won't fit the timing budget.

**Expected latency:** 30–50ms for ViT-S/14 on GPU (single forward pass, master precomputed). 60–90ms for ViT-B/14.

**Accuracy validation:** Run on all test images. For each image, record global_similarity, problem_patch_ratio, and save the similarity heatmap overlay.

For known-good pieces, global similarity should be > 0.90. Examine the heatmap — are there patches with low similarity that correspond to real locations (a stone, an engraving), or are they noise from alignment error or lighting variation? If good pieces routinely have low-similarity patches in specific locations (reflective facets, edges), those locations may need masking.

For decoration-defective pieces, verify that the defect location appears as a low-similarity region in the heatmap. Record whether the defect is detected by the global threshold, the patch ratio threshold, or both.

**GPU dependency check:** Measure the same stage on CPU. If CPU latency for ViT-S/14 exceeds 300ms, GPU is mandatory. Document this as a hardware requirement.

---

### Stage 7: Surface Quality Check

**What it does:** Detect surface defects (scratches, pits, tarnish) by comparing live surface texture against the master.

**Approach:** Compute local SSIM map between registered live and master grayscale images. Compute high-frequency difference using Laplacian filtering. Combine into a defect map. Threshold to get defect regions. Measure defect area ratio and count individual defect regions.

**Resolution consideration:** This check benefits from higher resolution. Run on a cropped ROI at original camera resolution (not the downsampled 1024×1024) for maximum sensitivity to fine scratches. The ROI is the bounding box of the piece from segmentation, extracted from the full-resolution image and registered using the same homography (scaled).

**Expected latency:** 40–60ms on CPU at 1024×1024 ROI. Scales with image area — if the full-res ROI is much larger, may need to tile.

**Accuracy validation:** Run on all test images. Save the defect heatmap and defect bounding boxes for every image. Visually verify on surface-defective images that the defect regions correspond to actual scratches/pits. On good pieces, verify that no spurious defect regions appear.

Key metrics per image: defect_ratio, num_defect_regions, max_defect_size, and the per-pixel defect map.

**Sensitivity tuning:** The scratch_sensitivity parameter and the SSIM window size control the tradeoff between catching fine scratches and generating false positives from alignment noise. Start with conservative settings (low sensitivity) and increase until you start catching your known defects, then stop. Document the setting that achieves zero false accepts.

---

### Stage 8: Decision Engine

**What it does:** Combine results from all three checks into a final ACCEPT / REVIEW / REJECT decision.

**Logic:**
- Registration confidence < threshold → REVIEW (don't trust any check results)
- Any check hard-fail → REJECT
- Any check borderline → REVIEW
- All checks pass → ACCEPT

**Expected latency:** < 1ms. Negligible.

**Accuracy validation:** Compare the system's decision against ground truth for every test image. Compute a confusion matrix (3×3: ACCEPT/REVIEW/REJECT predicted vs actual).

Primary metrics:
- **False Accept Rate (FAR):** defective pieces that received ACCEPT. Target: 0%.
- **False Reject Rate (FRR):** good pieces that received REJECT. Target: < 3%.
- **Review Rate on good pieces:** good pieces routed to REVIEW. Target: < 10%.
- **Review Rate on defective pieces:** defective pieces routed to REVIEW instead of REJECT. This is acceptable — REVIEW means a human catches it.
- **Escape Rate:** defective pieces that received ACCEPT. Must be 0% on the test set.

---

## 3. Visual Output — Defect Localization and Comparison Display

Every pipeline run must produce a visual comparison image that shows the operator (or the developer during validation) exactly where differences exist between master and live. This is not optional — an inspection system that says "FAIL" without showing where is unusable on a manufacturing floor.

### 3.1 Output Image Layout

For each inspected piece, generate a single composite output image with the following layout.

```
┌─────────────────────────────────────────────────────────────────┐
│  SKU: SKU_001  |  Decision: REJECT  |  Total Time: 247ms       │
├──────────────────┬──────────────────┬───────────────────────────┤
│                  │                  │                           │
│   MASTER IMAGE   │   LIVE IMAGE     │   DIFFERENCE OVERLAY      │
│   (reference)    │   (aligned)      │   (defects boxed)         │
│                  │                  │                           │
├──────────────────┴──────────────────┴───────────────────────────┤
│  CHECK RESULTS                                                  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  PROFILE    │  │ DECORATION  │  │  SURFACE    │             │
│  │  ✅ PASS    │  │ ❌ FAIL     │  │ ⚠️ REVIEW   │             │
│  │  IoU: 0.96  │  │ Sim: 0.72   │  │ Def%: 0.004 │             │
│  │             │  │             │  │             │             │
│  │ [contour    │  │ [similarity │  │ [defect     │             │
│  │  overlay]   │  │  heatmap]   │  │  heatmap]   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  TIMING: Pre 12ms | Seg 7ms | Rot 5ms | Reg 62ms |             │
│          Prof 24ms | Deco 49ms | Surf 53ms | Total 247ms       │
└─────────────────────────────────────────────────────────────────┘
```

The top row shows the master, the aligned live image, and a blended difference overlay side by side at equal scale. The bottom row shows per-check detail panels with visual evidence. The footer shows the per-stage timing breakdown inline.

### 3.2 Defect Bounding Boxes — How to Draw Them

Every detected difference between master and live must be bounded by a rectangle on the output image. The bounding box rules are specific to each check.

#### Profile Check Bounding Boxes

Compute the pixel-level difference between master silhouette mask and live silhouette mask. Find connected components in the difference mask (regions that exist in one but not the other). For each connected component larger than a minimum area threshold (to filter noise), compute the bounding rectangle. Draw the rectangle on the difference overlay in **red** with a 2px border. Label each box with the type of deviation: "EXCESS" if material exists in the live image but not the master (e.g., a bent prong sticking out), "MISSING" if material exists in the master but not the live image (e.g., a broken-off element).

On the live image panel, draw the master contour as a **green outline** and the live contour as a **yellow outline** so the shape deviation is immediately visible.

#### Decoration Check Bounding Boxes

From the DINOv2 similarity heatmap, threshold to find low-similarity regions (patches where cosine similarity falls below the per-check threshold). Upscale the patch-level mask to image resolution. Find connected components in the thresholded low-similarity mask. For each connected component, compute the bounding rectangle. Draw the rectangle on the live image in **orange** with a 2px border. Label each box with the similarity score of that region (e.g., "Sim: 0.61").

Additionally, overlay the full similarity heatmap as a semi-transparent color map on the difference overlay panel. Use a red-to-green color scale: green (>0.95 similarity) through yellow (0.85–0.95) to red (<0.85). This gives an at-a-glance view of where decorative elements match and where they diverge.

#### Surface Quality Bounding Boxes

From the combined defect map (SSIM + Laplacian), threshold to get binary defect regions. Find connected components. Filter out components smaller than the minimum defect size (20 pixels, to exclude noise). For each remaining component, compute the bounding rectangle. Draw the rectangle on the live image in **magenta** with a 2px border. Label each box with the defect area in pixels and a severity classification: "MINOR" if area < 100px, "MAJOR" if area > 100px.

For scratches specifically (defects with high aspect ratio — length >> width), draw a **rotated bounding box** that follows the scratch direction rather than an axis-aligned box. This provides a clearer visual for the operator.

Overlay the surface defect heatmap on the difference overlay panel using a blue-to-red color scale: blue (no anomaly) to red (strong anomaly).

### 3.3 Bounding Box Color Convention

Use consistent colors across all output images so operators and developers build visual intuition.

| Color | Meaning |
|-------|---------|
| Red (255, 0, 0) | Profile deviation bounding box |
| Green (0, 255, 0) | Master contour outline (reference) |
| Yellow (255, 255, 0) | Live contour outline (actual) |
| Orange (255, 165, 0) | Decoration mismatch bounding box |
| Magenta (255, 0, 255) | Surface defect bounding box |
| Cyan (0, 255, 255) | Registration keypoint matches (debug view only) |

### 3.4 Difference Overlay Panel — Blended View

The third panel in the top row (difference overlay) combines all three checks into one annotated image. Start with a 50/50 alpha blend of the master and aligned live image. This naturally highlights differences — matching regions look normal, differing regions show ghosting. On top of this blend, draw all bounding boxes from all three checks with their respective colors. This single panel gives a complete "where are the problems" view at a glance.

### 3.5 Output File Structure

Save all visual outputs to a structured directory per run.

```
outputs/
├── run_2026-05-25_143022/
│   ├── timing_dashboard.png
│   ├── timing_log.csv
│   ├── accuracy_report.csv
│   ├── results/
│   │   ├── SKU_001__good_001__ACCEPT.png
│   │   ├── SKU_001__scratch_001__REJECT.png
│   │   ├── SKU_001__missing_stone_001__REJECT.png
│   │   └── ...
│   ├── heatmaps/
│   │   ├── SKU_001__good_001__profile_diff.png
│   │   ├── SKU_001__good_001__decoration_heatmap.png
│   │   ├── SKU_001__good_001__surface_heatmap.png
│   │   └── ...
│   └── failures/
│       ├── SKU_001__borderline_001__FALSE_ACCEPT.png
│       └── ...
```

The filename encodes the SKU, image name, and decision so you can sort and filter results by outcome without opening each file. The `failures/` directory isolates misclassified images for quick review.

### 3.6 What the Bounding Boxes Must Validate

During accuracy benchmarking, the bounding boxes serve as evidence for defect localization, not just detection. For each known-defective image, manually verify two things: the bounding box correctly covers the actual defect location (localization accuracy), and no bounding boxes appear in regions where no defect exists (localization precision).

Record per-image: number of true positive boxes (box covers a real defect), number of false positive boxes (box on a non-defective region), and number of missed defects (real defect with no box). Aggregate these into localization precision and recall metrics alongside the per-check pass/fail metrics.

A system that correctly says "FAIL" but highlights the wrong region is dangerous — the operator will lose trust and start ignoring the system.

---

## 4. Speed Benchmarking Protocol

### 4.1 Timing Instrumentation

Every stage wraps its execution in a high-resolution timer. Use `time.perf_counter()` in Python — not `time.time()`, which has insufficient resolution. Log timing per image in a structured format.

```
image_path, stage, latency_ms
SKU_001/good/good_001.png, preprocess, 12.3
SKU_001/good/good_001.png, segmentation, 7.1
SKU_001/good/good_001.png, rotation_estimation, 4.8
SKU_001/good/good_001.png, fine_registration, 62.4
SKU_001/good/good_001.png, profile_check, 24.1
SKU_001/good/good_001.png, decoration_check, 48.7
SKU_001/good/good_001.png, surface_check, 53.2
SKU_001/good/good_001.png, decision, 0.2
SKU_001/good/good_001.png, total, 212.8
```

### 4.2 Visual Timing Display

Every pipeline run must produce a visual timing breakdown displayed to the user — not just logged to CSV. This serves two purposes: immediate feedback during development, and a presentable artifact for stakeholders.

#### Per-Image Timing Bar (Console Output)

After each image is processed, print a horizontal bar chart to the console showing time spent in each stage. This gives instant feedback on where time is going.

```
SKU_001/good/good_001.png — Total: 247.3ms [UNDER BUDGET]

  Preprocess       ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   12.3ms
  Segmentation     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    7.1ms
  Rotation Est.    █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    4.8ms
  Fine Registration████████████░░░░░░░░░░░░░░░░░░░   62.4ms
  Profile Check    █████░░░░░░░░░░░░░░░░░░░░░░░░░   24.1ms
  Decoration Check ██████████░░░░░░░░░░░░░░░░░░░░   48.7ms
  Surface Check    ██████████░░░░░░░░░░░░░░░░░░░░   53.2ms
  Decision         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.2ms
                   |         |         |         |
                   0ms     100ms     200ms     300ms
```

Color-code the total: green if under 500ms, yellow if 500–800ms, red if over 800ms. Mark individual stages red if they exceed their stage-level P95 hard limit.

#### Batch Summary Dashboard (After Full Test Run)

After processing the entire test set, generate a summary dashboard image (saved as PNG) that contains four panels.

**Panel 1 — Per-Stage Box Plot:** Box-and-whisker plot showing the distribution of latency for each stage across all images. Highlights outliers. Immediately reveals which stages have high variance.

**Panel 2 — Stacked Bar (Per Image):** Each image is a stacked bar showing the breakdown of time by stage. Images sorted by total time descending. Makes it obvious which images are slow and which stage is responsible.

**Panel 3 — Cumulative Timeline:** A waterfall/Gantt-style chart for a single representative image showing stages as sequential blocks on a timeline. Shows the critical path and where parallelization helps. Include both sequential and parallel versions side by side.

**Panel 4 — Budget Compliance:** A simple gauge or indicator showing what percentage of images completed under 500ms, under 800ms, under 1000ms, and over 1000ms. The primary go/no-go visual.

Save this dashboard to `outputs/timing_dashboard.png` after every batch run.

#### Timing Summary Table (Printed to Console)

After batch processing, print a formatted table with per-stage statistics.

```
┌─────────────────────┬──────────┬──────────┬──────────┬────────┐
│ Stage               │ Mean(ms) │ P95(ms)  │ Max(ms)  │ Status │
├─────────────────────┼──────────┼──────────┼──────────┼────────┤
│ Preprocess          │    11.2  │    14.8  │    18.3  │   ✅   │
│ Segmentation        │     6.8  │     9.1  │    12.4  │   ✅   │
│ Rotation Estimation │     5.1  │     7.3  │     9.8  │   ✅   │
│ Fine Registration   │    58.4  │    74.2  │    91.6  │   ✅   │
│ Profile Check       │    22.7  │    28.5  │    34.1  │   ✅   │
│ Decoration Check    │    46.3  │    52.8  │    67.4  │   ✅   │
│ Surface Check       │    51.8  │    63.2  │    78.9  │   ✅   │
│ Decision            │     0.2  │     0.3  │     0.4  │   ✅   │
├─────────────────────┼──────────┼──────────┼──────────┼────────┤
│ TOTAL (sequential)  │   202.5  │   250.2  │   312.9  │   ✅   │
│ TOTAL (parallel)    │   147.3  │   183.6  │   231.7  │   ✅   │
└─────────────────────┴──────────┴──────────┴──────────┴────────┘
Budget: 1000ms | Pass rate: 100% under budget
```

Status column shows ✅ if P95 is within the stage hard limit, ⚠️ if mean is within but P95 exceeds, ❌ if mean exceeds the hard limit.

### 4.3 What to Measure

For each stage, record mean latency, P95 latency (95th percentile — the "worst typical case"), and max latency across all test images.

P95 matters more than mean. A stage that averages 50ms but spikes to 400ms on certain images will blow the budget unpredictably. Identify which images cause spikes and why.

### 4.4 Warmup

Run the pipeline on 5 images before starting measurement. The first few runs incur cold-start costs (model loading, CUDA kernel compilation, CPU cache warming) that don't reflect production performance.

### 4.5 Sequential vs Parallel Timing

First measure all stages sequentially to get per-stage baselines. Then measure with the three checks running in parallel (using `concurrent.futures.ThreadPoolExecutor` or `ProcessPoolExecutor`) to get the real end-to-end time. Record both.

### 4.6 Hardware Documentation

Record the exact hardware used for benchmarking — CPU model, GPU model, RAM, storage type. Results are only meaningful tied to specific hardware. Include whether the GPU was under load from other processes.

### 4.7 Speed Acceptance Criteria

| Metric | Target | Hard Limit |
|--------|--------|------------|
| Mean total latency | < 500ms | < 800ms |
| P95 total latency | < 700ms | < 1000ms |
| Max total latency | < 1000ms | < 1500ms |
| Registration (stages 2-4) P95 | < 150ms | < 250ms |
| Any single check P95 | < 150ms | < 200ms |

If P95 exceeds the hard limit, identify the bottleneck stage and optimize before proceeding. Common fixes: reduce input resolution, limit SIFT keypoints, switch to ViT-S from ViT-B, or move a CPU stage to GPU.

---

## 5. Accuracy Benchmarking Protocol

### 5.1 Data Split

Split your labeled dataset into two sets. **Calibration set (70%):** used to tune thresholds for each check. **Test set (30%):** held out, never used for tuning. Final accuracy numbers are reported only on the test set.

Do not skip this split. Tuning and evaluating on the same data produces optimistic numbers that won't hold in production.

### 5.2 Threshold Calibration Procedure

For each check, the calibration process is the same.

1. Run the check on all calibration-set images and record the raw scores
2. Plot score distributions for PASS and FAIL ground-truth images separately
3. Set the REJECT threshold at the point where false accept rate reaches 0% on the calibration set
4. Set the REVIEW threshold at the point where the remaining false reject rate drops below 5%
5. The gap between REJECT and REVIEW thresholds defines the "borderline" zone routed to human review
6. Document both thresholds and the calibration-set metrics at those thresholds

### 5.3 Per-Check Metrics

For each check independently, compute on the held-out test set:
- True Positive Rate (defects correctly caught)
- False Positive Rate (good pieces incorrectly flagged)
- Precision and Recall at the operating threshold
- The score distribution plots for pass/fail

### 5.4 End-to-End Metrics

For the combined system decision, compute on the test set:
- False Accept Rate (must be 0% on test set)
- False Reject Rate (target < 3%)
- Review Rate (target < 15%)
- Per-SKU breakdown of all metrics (some SKUs will be harder)
- Per-defect-type detection rate (which defects are caught, which are missed?)

### 5.5 Failure Analysis

For every image where the system decision disagrees with ground truth, document the image path, the expected vs actual decision, which check(s) were wrong, the raw scores from each check, and the visual evidence (save the diff masks, heatmaps, overlays).

This failure log is the most valuable output of the validation. It tells you exactly where to improve — whether the problem is imaging, registration, thresholds, or a fundamental limitation of the approach.

---

## 6. Experiment Execution Plan

### Experiment 1: Registration Reliability

**Goal:** Validate that rotation estimation + SIFT alignment works across orientations.

**Setup:** Take 1 SKU. Photograph the same good piece at 8 orientations (0°, 45°, 90°, ..., 315°). Run stages 1–4 on each image.

**Success criteria:** Registration NCC > 0.85 on all 8 images. Rotation error < 3° on all images. Visual overlay confirms correct alignment.

**If this fails:** Do not proceed to the three checks. Fix rotation estimation or improve imaging setup first.

### Experiment 2: Profile Check Baseline

**Goal:** Validate profile check catches shape defects and passes good pieces.

**Setup:** Use the registered images from Experiment 1 (good pieces) plus profile-defective pieces for 1 SKU.

**Success criteria:** 0% false accept, < 5% false reject on good pieces.

### Experiment 3: Decoration Check Baseline

**Goal:** Validate DINOv2 feature comparison catches decoration defects.

**Setup:** Same SKU. Good pieces + decoration-defective pieces. Run the decoration check and record similarity heatmaps.

**Success criteria:** All decoration defects produce global_similarity below 0.90. All good pieces score above 0.90. Heatmaps localize the defect correctly.

**Also measure:** ViT-S vs ViT-B accuracy difference. If both achieve 0% false accept, use ViT-S for the speed advantage.

### Experiment 4: Surface Check Baseline

**Goal:** Validate SSIM + Laplacian comparison catches surface defects.

**Setup:** Same SKU. Good pieces + surface-defective pieces.

**Success criteria:** All surface defects detected. Scratch sensitivity parameter tuned to catch the finest scratch in the test set without triggering false positives on good pieces.

### Experiment 5: End-to-End Pipeline

**Goal:** Run the full pipeline on all SKUs, all images. Measure speed and accuracy together.

**Setup:** Full test dataset. All stages. Timing instrumentation active.

**Deliverables:** Per-stage timing report (mean, P95, max). End-to-end confusion matrix. Per-SKU accuracy breakdown. Failure analysis log. Go/no-go recommendation for production pilot.

---

## 7. Deliverables from Validation

At the end of validation, you should have the following.

1. **Timing report:** Per-stage latency breakdown with mean/P95/max, confirming total pipeline fits within 1 second on target hardware. Includes the timing dashboard PNG with box plots, stacked bars, waterfall chart, and budget compliance gauge.

2. **Accuracy report:** Confusion matrix, FAR, FRR, review rate on the held-out test set. Per-SKU and per-defect-type breakdown.

3. **Threshold configuration file:** Calibrated thresholds for each check, documented with the calibration data that produced them.

4. **Failure log:** Every misclassified image with root cause analysis and visual evidence.

5. **Hardware specification:** Exact camera, GPU, CPU requirements validated by benchmarking.

6. **Visual inspection results:** Per-image composite output images showing master vs aligned live vs difference overlay, with color-coded bounding boxes around every detected defect, per-check heatmaps, and inline timing breakdown. All saved to the structured output directory.

7. **Localization accuracy report:** Per-image count of true positive, false positive, and missed defect bounding boxes. Aggregate localization precision and recall. Evidence that the system not only detects defects but correctly shows where they are.

8. **Go/no-go decision:** Based on whether FAR = 0%, total pipeline P95 < 1 second, and defect localization recall > 90%. If any criterion is not met, the report identifies which stage is the bottleneck and what the remediation options are.

---

## 8. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Registration fails on symmetric pieces | False accepts from wrong alignment | Medium | Test with symmetric SKUs early (Experiment 1). Add hallmark detection if needed. |
| DINOv2 inference exceeds budget on target GPU | Pipeline exceeds 1 second | Low | Benchmark ViT-S first. Have ONNX/TensorRT export ready as fallback. |
| Surface check triggers on alignment artifacts at edges | High false reject rate | High | Erode comparison mask by 5–10px. Validate in Experiment 4. |
| Good pieces with natural variation flagged as defective | High review rate, operator fatigue | Medium | Collect more "acceptable variation" examples. Widen thresholds on calibration set. |
| SIFT fails on featureless polished surfaces | Registration unreliable for plain bands | Medium | Test plain bands in Experiment 1. Fallback to contour-based alignment. |
| Lighting variation between master capture and live capture | Systematic bias in all checks | High | Capture master and live on same station, same settings. Validate with images taken days apart. |
