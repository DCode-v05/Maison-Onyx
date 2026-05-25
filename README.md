# Jewelry Visual Inspection

Implementation of the 8-stage reference-based inspection pipeline described in
[jewelry_inspection_validation_plan.md](jewelry_inspection_validation_plan.md),
fronted by a React UI that mirrors the "onyx & gold" look of the original
Streamlit prototype.

## Layout

```
backend/                    FastAPI service + 8-stage pipeline
  main.py                   API endpoints (/api/info, /api/inspect, /api/healthz)
  pipeline/
    preprocess.py           Stage 1
    segmentation.py         Stage 2
    rotation.py             Stage 3
    registration.py         Stage 4  (SIFT + RANSAC homography)
    profile_check.py        Stage 5  (Hu moments, area dev, silhouette IoU)
    decoration_check.py     Stage 6  (DINOv2 patch similarity)
    surface_check.py        Stage 7  (SSIM + Laplacian)
    decision.py             Stage 8
    visualizer.py           Bounding boxes, heatmap colormaps, overlay
    orchestrator.py         Runs all stages with per-stage timing
    types.py                Shared dataclasses

frontend/                   Vite + React + TypeScript UI
  src/
    App.tsx, main.tsx
    api.ts, types.ts
    components/             Masthead, Hero, UploadRow, Verdict,
                            ComparisonView, CheckResults, TimingRow, Footer
    styles/app.css          Ported tokens, typography, components from app.py
```

## Backend

Python ≥ 3.10. Install dependencies (torch separately if you need CUDA):

```
pip install -r requirements.txt
# CUDA wheel example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Run the API:

```
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET  /api/healthz`     liveness
- `GET  /api/info`        model + pipeline info for the Instrument panel
- `POST /api/inspect`     multipart: `master`, `live` → full inspection result

## Frontend

```
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxies /api to :8000)
```

The dev server proxies `/api` to `http://localhost:8000` (configured in
`vite.config.ts`).

## Acceptance reminders

From the validation plan:
- Total pipeline P95 < 1000 ms
- 0% false accept on the test set
- Defect localization recall > 90%
