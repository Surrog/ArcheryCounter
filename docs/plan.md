# ArcheryCounter — Implementation Plan

See `docs/research.md` for design rationale.

---

## Status

| Phase | Description | Status |
|---|---|---|
| P1–P9 | Boundary, calibration, ring detection, annotation tooling, rule-based arrow detection | ✅ |
| P10 | Scoring pipeline | ~~discarded — NN score head handles scoring~~ |
| P11 | Neural network arrow tip detector (MobileNetV2 + FPN, tip + score heads) | 🔄 in progress |
| P12 | NN integration into mobile app | ⬜ |

---

## Pipeline (current)

```
Photo
  → [P1] Boundary polygon (4–8 vertices), clamped to image bounds
  → [P2] ColourCalibration (per-image HSV references per zone)
  → [P3+P4] Colour-guided ring detection (32 rays × 10 boundaries)
  → SplineRing[10] (K=12 Catmull-Rom control points per ring)
Output: { rings, paperBoundary, calibration, ringPoints }
```

Arrow detection pipeline (P11, current):

```
Photo
  → [P1–P4] Ring detection (above)
  → [P11] NN inference: letterbox 640×640 RGB → ONNX → heatmap NMS → ScoredArrow[{ tip, score }]
Output: { rings, paperBoundary, arrows[{ tip, score }] }
```

---

## Phase 11 — Neural Network Arrow Tip Detector

**Architecture:** MobileNetV2 backbone + 3-level FPN + tip heatmap head + score classification head.
- Input: (3, 640, 640) — letterboxed RGB (radial-distance channel dropped; see M6 in `docs/nn_plan.md`)
- Output: tip heatmap (1, 160, 160) + score logits (11, 160, 160) — score read at detected tip location
- Training: focal loss (tip) + cross-entropy (score) + sparsity penalty λ=20; AdamW + cosine annealing 60 epochs
- Checkpoint selection by best val F1; top-50 prediction cap prevents dense heatmaps from gaming recall

**Training infra** (`scripts/nn/`): `dataset.py`, `model.py`, `train.py`, `export.py`, `eval-nn.ts`

- [x] **P11-T1** Dataset, model, training loop, ONNX export
- [x] **P11-T2a/b** `onnxruntime` packages added; `src/arrowDetector.ts` implemented
- [x] **P11-T2e** `scripts/eval-nn.ts` — offline recall evaluation
- [x] **P11-T2c** Wire `detectArrowsNN` into `ArcheryCounter.processImage`; surface `score` from NN output; returns empty arrows if ONNX unavailable (no rule-based fallback)
- ~~**P11-T2f**~~ INT8 quantisation produces wrong detections (calibration broken at 640×640); using FP32 (12 MB)

---

## Phase 12 — Mobile App Integration

### Strategy

The ONNX model runs via `onnxruntime-react-native` (already in `package.json`). The model file must be bundled as a static asset.

**iOS:**
1. Add `arrow_detector_int8.ort` to the Xcode project under `ios/ArcheryCounter/` and mark as "Copy Bundle Resources".
2. At runtime resolve with `require('../assets/arrow_detector_int8.ort')` (Metro bundler) or `RNFS.MainBundlePath + '/arrow_detector_int8.ort'` (react-native-fs).
3. `onnxruntime-react-native` loads from a file-system path; pass the resolved path to `detectArrowsNN`.

**Android:**
1. Place `arrow_detector_int8.ort` in `android/app/src/main/assets/`.
2. Resolve with `RNFS.DocumentDirectoryPath` after copying from assets on first launch, or use the `asset://` URI scheme supported by onnxruntime-react-native.

**Size:** FP32 ONNX ≈ 12 MB. INT8 quantisation currently broken (calibration fails at 640×640 input); use FP32.

### Tasks

- [ ] **P12-T1** Fix INT8 quantisation (recalibrate at 640×640) or ship FP32; convert to `.ort` format
- [ ] **P12-T2** Add model file to iOS bundle resources and Android assets
- [ ] **P12-T3** `src/arrowDetector.ts` — resolve model path per platform (`Platform.OS`)
- [ ] **P12-T4** Measure on-device latency (target: < 2 s on iPhone 12); if slow, reduce input to 384×384 and retrain

---

## Key design decisions

- Ring index 0 = innermost (bullseye), index 9 = outermost. Score 10 = bullseye, 1 = outermost, 0 = miss.
- NN input includes a radial-distance channel computed from rule-based rings — gives the network explicit geometry without learning it from colour alone.
- Arrow detection precision-first: missing arrows acceptable, false positives are not.
