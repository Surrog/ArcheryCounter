# NN Improvement & Experimentation Plan

Companion to `docs/plan.md` §P11–P12. Tracks training infrastructure, model, and data improvements independently of the main product roadmap.

---

## Status snapshot

| Metric | Value |
|---|---|
| Training images | 262 (242 train + 20 val) |
| Input | 3ch RGB, 640×640 (radial channel dropped — see M6) |
| Sparsity weight | λ=20 |
| Best val F1 (latest run, in progress) | ~0.150 @ epoch 17 |
| Best val recall@45px (latest run) | ~0.373 @ epoch 17 |
| Main remaining bottleneck | Minority domain coverage (WA, 2019 images); see A3/A4 |

---

## D — Data pipeline

### D1 — RAM image cache ✅ (superseded by D2)
Pre-load all images into Python memory at `ArrowDataset.__init__`. Eliminates disk I/O during training.

- **When:** dataset fits in RAM (< ~500 images at 1200px ≈ ~1.5 GB)
- **Implementation:** cache is `{fname: (img_np, radial, scale, pad_x, pad_y, orig_w, orig_h)}` — letterboxed uint8 numpy + radial channel pre-computed at init; `__getitem__` does `.copy()` before augmentation
- **Actual gain:** ~24 epochs in 2 min (was ~26 epochs in prior session with much slower epochs)

### D2 — LRU cache ✅
Bounded per-worker in-memory cache. Replaced D1's full pre-load after 640×640 upgrade caused ~2.5 GB RAM pressure (5 processes × 500 MB forked copies).

- **Implementation:** module-level `@lru_cache(maxsize=80)` on `_load_letterboxed(path)` — each worker builds its own LRU lazily (~96 MB/worker × 4 workers = ~384 MB). Radial channel recomputed per sample (~1 ms, negligible).

### D5 — Dense false-positive diagnosis + training fixes ✅
Diagnostic showed model produced 713 NMS peaks per image (all at ~0.999 confidence); GT peaks suppressed because neighbouring pixels were equally hot. Root cause: the focal loss's pos term drives the bias up 35× faster than the neg term can counteract it (separate pos/neg normalisation with n_pos≪n_neg).

Three fixes applied to `train.py`:
1. **Sparsity loss** `λ * tip_hm.mean()` with λ=5.0 — balances the bias equilibrium to mean≈0.10 (vs ≈0.43 without)
2. **F1 as checkpoint metric** — prevents dense-prediction checkpoints (recall=0.9 but precision=0.7%) from being saved as "best"
3. **Top-50 prediction cap + precision tracking** — validate_recall now returns (recall, precision, F1, avg_pred_count)

---

### D3 — LMDB pre-decoded store
One-time build script decodes all images to letterboxed uint8 numpy arrays and stores in LMDB (memory-mapped). Workers read with ~5µs zero-copy access.

- **When:** > 5k images
- **Files:** `scripts/nn/build_lmdb.py` (build), dataset reads from LMDB instead of JPEG
- **Storage:** ~7.5 GB per 10k images at 512×512
- [ ] Implement `build_lmdb.py`
- [ ] Add LMDB read path to `dataset.py` (toggled via `--lmdb` flag)

### D4 — turbojpeg decode
Replace PIL JPEG decode with libjpeg-turbo (`python-turbojpeg`). 3–5× faster decode, same output.

- **When:** intermediate dataset sizes where LMDB isn't justified yet
- [ ] Implement as a drop-in in `dataset.py`

---

## T — Training loop

### T1 — Increase batch size + reduce workers ✅
With D1 in place, workers are CPU-bound on augmentation only. Fewer workers = less IPC overhead; larger batch = better GPU utilisation.

- New defaults: `--batch 32 --workers 4`

### T2 — `persistent_workers` + `prefetch_factor` ✅
Eliminates worker restart overhead between epochs and keeps the prefetch queue deeper.

- Implemented: `persistent_workers=True, prefetch_factor=4` (guarded by `num_workers > 0`)

### T3 — Mixed precision (`torch.autocast`) ✅
MPS supports float16 for most ops. ~2× throughput on matrix ops, half the memory.

- Implemented: `torch.autocast(device_type=device, enabled=(device != 'cpu'))` wraps forward + loss computation

### T4 — `torch.compile` ❌ blocked
Fuses ops for MPS. Tested with `fullgraph=False` — fails at first forward pass with Metal shader compilation error (undeclared identifier in generated MSL code). PyTorch MPS support not mature enough.

- Commented out in `train.py`; revisit with future PyTorch version

### T5 — GPU augmentation
Move colour jitter + flips to GPU using `torchvision.transforms.v2` after tensor conversion. Removes CPU→GPU handoff wait.

- **When:** augmentation becomes the bottleneck (profile first)
- [ ] Profile CPU augmentation share of epoch time before investing

---

## M — Model

### M1 — Larger input resolution ✅
Current: 512×512. Arrows at long shooting distances are very small (< 10px tip).

- Result: 640×640 → recall@45 **0.905** vs 0.740 (+0.165). Kept.

### M2 — Stronger backbone ✅
MobileNetV2 is fast but weak on small objects. Tried MobileNetV3-Large.

- `--backbone mobilenet_v2|mobilenet_v3_large` added to `train.py`
- Result: MobileNetV3-Large = **0.861** vs MobileNetV2 **0.905** (-0.044). Reverted to V2. V3 SE modules + hard-swish harder to fine-tune on small dataset.

### M3 — Deeper FPN / larger FPN channels ✅
Current: FPN_CH=128. Richer features may help on cluttered backgrounds.

- Result: FPN_CH=256 → recall@45 **0.691** vs 0.905 (-0.214). Reverted. Dataset too small to benefit; model overfit.

### M4 — Smaller Gaussian sigma ✅
Current SIGMA=3.0 (heatmap pixels). Tighter targets force more precise localisation.

- Result: SIGMA=2.0 → recall@45 **0.740** vs baseline 0.682 (+0.058). Kept.

### M6 — Input channel ablation
Verify that heatmap output format and the radial-distance input channel each genuinely contribute to recall, rather than hurting or being neutral.

Three conditions to train and compare on the same data split:

| Condition | Input | Notes |
|---|---|---|
| RGB only | 3ch (RGB) | Drop radial channel; `model.stage1` input_channels=3 |
| Radial only | 2ch (grey + radial) | Replace RGB with luminance; test if geometry alone is sufficient |
| RGB + radial | 4ch (current) | Baseline |

- ✅ `--input-channels 3|4` added to `train.py`; `ArrowDetector(in_channels=...)` adjusts first conv
- Result: RGB-only (3ch) = **0.858**, RGB+radial (4ch) = **0.905** → radial channel worth **+0.047**.
- **Decision (2026-04-01):** Switched to RGB-only (3ch). Radial channel requires ring detection at inference — but `eval-nn.ts` passes empty rings, causing a train/eval mismatch. −0.047 recall accepted to eliminate this dependency. `arrowDetector.ts` updated: `fillRadialChannel` removed, tensor shape `[1,3,640,640]`.

### M5 — Multi-scale inference
Run inference at 3 scales (0.75×, 1×, 1.25×), average heatmaps before NMS. Helps for targets far/close to camera.

- [ ] Implement in `arrowDetector.ts` and `eval-nn.ts`; measure recall gain

---

## A — Annotation & data quality

### A1 — Annotate WA images ✅
`IMG-20260330-WA*` images annotated. Still near-0 recall due to domain imbalance → see A3/A4.

### A2 — Annotate 2019 images ✅
2019 images annotated. Still near-0 recall due to domain imbalance → see A3/A4.

### A3 — Weighted random sampling for minority domains
WA and 2019 images are annotated but too few to compete with the majority. Over-sample them so the model sees each minority image as often as N majority images.

- **Implementation:** compute per-sample weight at `ArrowDataset.__init__`; pass `WeightedRandomSampler` to `DataLoader` (drop `shuffle=True`)
- **Heuristic:** start at 5× weight for minority domains; tune up/down based on whether minority recall improves without majority recall dropping
- [ ] Identify minority filenames (prefix-based: `IMG-`, `2019`)
- [ ] Implement `WeightedRandomSampler` in `train.py`
- [ ] Expose `--minority-weight` CLI arg (default 5.0)
- [ ] Retrain and compare per-domain recall

### A4 — Domain-targeted augmentation
Stronger augmentation on minority images synthesises more variety from few examples. First, visually inspect WA vs majority images to identify the actual difference (scale, compression, colour cast).

- **WA images** (WhatsApp-compressed): likely smaller apparent arrow size → widen scale range to `(0.4, 1.3)`, add JPEG compression augmentation (`ImageCompression(quality_lower=50)`)
- **2019 images**: different camera/target → stronger hue/saturation jitter, possibly add `Sharpen`
- **Implementation:** per-sample augmentation pipeline selected by filename; or increase global augmentation strength and rely on A3 sampling to focus it on minority images
- [ ] Visual inspection: compare 3–5 WA and 2019 images vs typical training images
- [ ] Add `ImageCompression` augmentation for WA images
- [ ] Extend scale range globally or per-domain
- [ ] Retrain after A3 is in place; measure delta

### A5 — Pseudo-labeling (semi-supervised)
Run rule-based pipeline on unannotated images; keep detections where `success === true` and ring count = 10. Use as weak training signal with 0.5× loss weight.

- [ ] `scripts/nn/pseudo_label.py` — generate pseudo-labels from rule-based pipeline
- [ ] Add `pseudo_weight` param to `focal_loss`
- [ ] Train with pseudo-labels; compare recall@45px

---

## E — Export & deployment

### E1 — INT8 export ❌ blocked
Attempted static INT8 quantisation via onnxruntime — produces wrong detections at 640×640 (all peaks cluster in top strip of heatmap). Calibration reader likely fails to produce a representative distribution at the new input size.

- Currently using FP32 (12 MB) everywhere
- [ ] Diagnose: run calibration with explicit 640×640 inputs; check per-layer quantisation error
- [ ] Re-attempt once root cause is understood

### E2 — `.ort` format for React Native
`onnxruntime-react-native` prefers the pre-optimised `.ort` format over raw `.onnx`.

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort arrow_detector_int8.onnx
```
- [ ] Convert and test loading in RN environment

### E3 — On-device latency measurement (P12-T5)
Target: < 2 s on iPhone 12.

- [ ] Measure FP32 latency
- [ ] Measure INT8 latency
- [ ] If slow: reduce input to 384×384 and retrain

---

## Experimentation log

| Date | Change | Recall@45 (train) | Recall@45 (eval) | Notes |
|---|---|---|---|---|
| 2026-03-31 | Baseline (54 images, FP32, SIGMA=3) | 0.775 | 0.222 | Coord fix applied |
| 2026-03-31 | +39 images (93 total) | 0.758 | 0.217 | More GT, harder val set |
| 2026-03-31 | +18 more images (181 total), D1+T1-T3, batch=32, patience=10 | 0.705 (ep14) | — | Early stopping active; ~24 ep/2min |
| 2026-03-31 | M4: SIGMA=2.0 | 0.740 | — | +0.058 vs baseline; kept |
| 2026-03-31 | M1: 640×640 input | 0.905 | — | +0.165 vs M4; kept |
| 2026-03-31 | M3: FPN_CH=256 | 0.691 | — | -0.214 vs M1; reverted |
| 2026-03-31 | M6: RGB-only (3ch) | 0.858 | — | radial channel worth +0.047 |
| 2026-03-31 | M2: MobileNetV3-Large | 0.861 | — | -0.044 vs V2; reverted |
| 2026-03-31 | Fix: F1 metric + λ=5 sparsity loss | F1=0.132 (ep11) | recall=0.705 | Prior 0.905 "recall" was spurious (713 dense FP per image); λ=5 stabilises bias at mean≈0.10 |
| 2026-04-01 | Switch to 3ch RGB (no radial channel) | F1=0.150 (ep1 best), recall@ep25=0.578 | — | Eliminates train/eval mismatch; radial channel at inference was all zeros. Deployed last.pt (recall=0.578, prec=0.068). F1 metric picks undertrained ep1 — consider recall-weighted checkpoint. |
| 2026-04-09 | λ=20 sparsity, 262 images (new annotations), 60 ep | F1≈0.150 @ ep17 (in progress) | — | Stronger sparsity penalty to reduce false positives; more training data |
