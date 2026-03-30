# ArcheryCounter — Implementation Plan

See `docs/research.md` for design rationale.

---

## Status

| Phase | Description | Status |
|---|---|---|
| P1 | Boundary detection (ray-cast + polygon) | ✅ |
| P2 | Per-image colour calibration (von Kries white-balance) | ✅ |
| P3 | Colour-guided radial ring detection (32 rays, 10 boundaries/ray) | ✅ |
| P4 | White ring / outer boundary (regression extrapolation) | ✅ |
| P5 | Annotation tool migration to spline control points | ✅ |
| P6 | Algorithm output migrated to `SplineRing[]` | ✅ |
| P7 | Ground-truth test suite (PostgreSQL annotations) | ✅ |
| P8 | Arrow annotation in annotate tool | ✅ |
| P9 | Arrow detection algorithm (rule-based Hough, recall≈0.70) | ✅ |
| P10 | Scoring pipeline | ⬜ |
| P11 | Neural network arrow tip detector | ⬜ next |

---

## Pipeline (current)

```
Photo
  → [P1] Boundary polygon (4–8 vertices)
  → [P2] ColourCalibration (per-image HSV references per zone)
  → [P3+P4] Colour-guided ring detection (32 rays × 10 boundaries)
      ↳ R2 ratio clamp: ring[7] rebuilt from ring[5]×8/6 if r7/r5 out of [1.05, 1.65]
      ↳ white rings [8,9] extrapolated via OLS regression through [1,3,5,7]
  → SplineRing[10] (K=12 Catmull-Rom control points per ring)
  → [P9] Arrow detection (zone-adaptive dark-pixel Hough within outermost ring)
Output: { rings, paperBoundary, calibration, ringPoints, arrows }
```

---

## Phase 10 — Scoring pipeline

See `docs/research.md §5` for the full design.

- [ ] **P10-T1** `splineCentroid(ring): [number, number]` and `splineRadius(ring): number` helpers (mean distance from centroid to control points) — add to `src/spline.ts`
- [ ] **P10-T2** `samplePatchZone(rgba, width, height, tip, cal): ZoneName | null` — modal zone of annular patch (inner r=4, outer r=12), excludes hay pixels (S < 0.15 or yellow-low-sat), in `src/scoring.ts`
- [ ] **P10-T3** `scoreArrow(tip, rings): number | 'X'` — walk rings[0..9] with `pointInClosedSpline`; X-ring check via `dist < 0.4 × splineRadius(rings[1])`, in `src/scoring.ts`
- [ ] **P10-T4** `scoreArrowWithCheck(rgba, width, height, tip, rings, cal): ScoredArrow` — calls T3 + T2 cross-check; sets `lowConfidence` if colour zone disagrees, in `src/scoring.ts`
- [ ] **P10-T5** Export `ScoredArrow` type from `src/scoring.ts`; update `ArrowDetection` in `arrowDetection.ts` to add optional `score` field
- [ ] **P10-T6** Wire into `ArcheryCounter.processImage`: after `findArrows`, call `scoreArrowWithCheck` for each arrow; include scores in `ProcessImageResult`
- [ ] **P10-T7** Add scoring assertions to `src/__tests__/groundTruth.test.ts`: strict equality where tip > 10 px from nearest ring boundary; ±1 tolerance near boundaries

---

---

## Phase 11 — Neural Network Arrow Tip Detector

**Motivation:** The rule-based Hough pipeline plateaus at ~70% recall due to fundamental failure modes (collinear arrows, short shafts in masked zones, straw-bale FPs). A trained network learns lighting-invariant shaft features from data rather than hand-crafted thresholds.

**Scope:** Tip-position detection + score in image coordinates. Ring detection (P1–P4) stays rule-based (ring IoU ≈ 0.92 is already good); the NN replaces P9 only.

**Architecture:** CenterNet-style heatmap on MobileNetV2 backbone.
- Input: 512×512 RGB, full image resized with letterboxing (no crop — paper boundary detection is too inaccurate to crop reliably).
- Backbone: MobileNetV2 (ImageNet pretrained, freeze first 7 layers initially).
- Neck: 3-layer FPN (channels 32/64/128) with bilinear upsample.
- Head: 1×1 Conv → 128×128 single-channel heatmap (Gaussian peak at each tip, σ=3 px in heatmap space).
- Loss: Focal loss (α=2, β=4) on heatmap.
- Export: ONNX → INT8 quantised `.ort` for `onnxruntime-react-native` (~4 MB).

**Training data:** 54 annotated images → 273 arrows. With augmentation (colour jitter ±40%, rotation ±20°, horizontal/vertical flip, scale ±25%, perspective warp, Gaussian blur σ=0–2) effective dataset ≈ 2 000–5 000 crops. Semi-supervised expansion via pseudo-labels from the existing pipeline is planned (P11-T5).

**Post-processing:** NMS on 128×128 heatmap (kernel=5, threshold=0.35) → tip list in heatmap space → rescale to original image coordinates. Score via existing geometric `scoreArrow` (P10-T3) using rule-based rings.

---

### P11-T1 — Python training script `scripts/nn/train.py`

Data pipeline:
- Connect to PostgreSQL (`psycopg2`); query `annotations` table for `filename`, `arrows`.
- Load images from `images/` with Pillow; letterbox-resize to 512×512 (pad with black, preserve aspect ratio, no crop).
- Track the letterbox scale + offset so tip coordinates map correctly to 128×128 heatmap space.
- Convert annotated tips to 128×128 Gaussian heatmaps (sigma=3 px in heatmap space).
- PyTorch `Dataset` + `DataLoader` (batch=8, `num_workers=4`).
- Augmentation with `albumentations`: `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `ShiftScaleRotate(scale=0.25, rotate=20)`, `ColorJitter(brightness=0.4, hue=0.1)`, `GaussianBlur`, `Perspective`.

Model:
- `torchvision.models.mobilenet_v2(pretrained=True)` as feature extractor.
- FPN neck merging features from layers 4, 7, 14 of MobileNetV2.
- Heatmap head: Conv(256→128→1) + sigmoid.

Training:
- AdamW, lr=1e-3, weight decay=1e-4; cosine annealing 60 epochs.
- Focal loss: `((1-p)^α * -log(p))` at GT peaks, `p^β * -log(1-p)` elsewhere.
- Validation: 10% hold-out; metric = recall@45px on original image coordinates.
- Save best checkpoint as `scripts/nn/arrow_detector.pt`; export `scripts/nn/arrow_detector.onnx`.

Outputs: `arrow_detector.onnx`, `train_log.csv`, `val_predictions_epoch_N.jpg` (visual check every 10 epochs).

```
scripts/nn/
  train.py          ← main training script
  model.py          ← MobileNetV2 + FPN + heatmap head
  dataset.py        ← PostgreSQL → PyTorch Dataset
  export.py         ← PT → ONNX → INT8 quantisation
  requirements.txt  ← torch, torchvision, albumentations, psycopg2, onnx, onnxruntime
```

- [ ] **P11-T1a** `dataset.py` — PostgreSQL loader, crop-to-boundary, heatmap generation, augmentation pipeline
- [ ] **P11-T1b** `model.py` — MobileNetV2 backbone + FPN neck + heatmap head
- [ ] **P11-T1c** `train.py` — training loop, focal loss, validation metric, checkpoint saving
- [ ] **P11-T1d** `export.py` — export `.pt` → `.onnx`; INT8 static quantisation; sanity-check ONNX output vs PyTorch

---

### P11-T2 — TypeScript inference integration

Replace `findArrows` call in `ArcheryCounter.processImage` with ONNX inference.

- [ ] **P11-T2a** Add `onnxruntime-react-native` to `package.json`; bundle `arrow_detector.ort` as a static asset.
- [ ] **P11-T2b** `src/arrowDetector.ts` — `detectArrowsNN(rgba, width, height): ArrowDetection[]`
  - Letterbox-resize rgba to 512×512 (same transform as training, no crop).
  - Run ONNX session; get 128×128 heatmap.
  - NMS on heatmap (kernel=5, threshold=0.35) → peaks.
  - Map peaks back to original image coordinates via inverse letterbox transform.
  - Return `ArrowDetection[]` with `tip` in original coords; `nock` is null (NN does not predict it).
- [ ] **P11-T2c** Wire into `ArcheryCounter.processImage`: call `detectArrowsNN` instead of `findArrows`; feed detections into `scoreArrowWithCheck` (P10-T4).
- [ ] **P11-T2d** Fallback: if ONNX runtime unavailable (e.g. web preview), fall back to `findArrows` (rule-based).
- [ ] **P11-T2e** Evaluation script `scripts/nn/eval.ts` — same greedy bipartite TP/FN metric as `accuracy-report.ts` but against the NN output; run on held-out images.

---

### P11-T3 — Semi-supervised expansion (optional, after T1 baseline)

- [ ] Run current rule-based pipeline on all unnanotated images; keep detections where `success === true` and detected ring count = 10.
- [ ] Pseudo-label tip positions; add to training set with lower weight (0.5×) in focal loss.
- [ ] Re-train; compare recall@45px before/after.

---

### P11 — Planned additional features

| # | Feature | Notes |
|---|---|---|
| 1 | **Nock-endpoint auxiliary head** | Second heatmap head (same FPN neck) for nock positions. Shaft direction from tip→nock enables post-processing that suppresses detections with implausible angles (e.g. tangential to rings). Train with same focal loss; nock GT from annotated `nock` field. |
| 2 | **Score classification head** | Small MLP attached at each heatmap peak position (bilinear-sampled FPN features) → 11-class softmax (0=miss, 1–10). Removes dependence on ring detection for scoring; exposes score uncertainty as a probability vector. |
| 3 | **Radial distance input channel** | Third input channel = distance from target centre normalised by outermost ring radius (0 at centre, 1 at outermost ring, clamped). Computed from rule-based ring output before inference; gives the network explicit radial geometry without learning it from colour. |
| 4 | **Multi-scale inference** | Run inference at 3 scales (0.75×, 1×, 1.25× of the 512×512 letterboxed input); upsample all three 128×128 heatmaps to a common size and average before NMS. Helps for very small or large targets relative to the image frame. |

---

## Key design decisions

- Ring index 0 = innermost (bullseye), index 9 = outermost. Score 10 = bullseye, 1 = outermost, 0 = miss.
- `BOOTSTRAP_SCALE = 2`: pretreatment runs on 2× downsampled image; centroids/radii scaled back.
- `N_BOUNDARY = 180`, `N_RINGS = 32`.
- Monotonicity enforcement in `detectRingDistancesOnRay`: forward pass on `transitionDist[]` + final pass on `result[0..9]`. Both nullify violations; missing rings filled by interpolation on neighbouring rays.
- Arrow detection precision-first: missing arrows acceptable, false positives are not.
