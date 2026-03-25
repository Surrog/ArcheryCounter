# Detection Performance Report

Generated from 25 annotated images (10 from 2019-03 batch, 15 from 2026-03-19 batch).
Benchmark script: `scripts/benchmark.ts` — reads `annotations` vs `generated` tables.
Ring metric: symmetric mean boundary distance (px). Arrow match threshold: 40 px.

---

## Summary

| Metric | 2019 batch (10 imgs) | 2026-03-19 batch (15 imgs) | Overall |
|---|---|---|---|
| Detection failures | 0/10 | 0/15 | 0/25 |
| Rings 0–5 mean error | 2.1 px | 0.5 px* | — |
| Rings 6–8 mean error | 3.1 px | **69 px** | — |
| Ring 9 mean error | 4.9 px | 6.0 px | — |
| Arrow recall | 77 % | 46 % | 60 % |
| Arrow precision | 77 % | 21 % | 36 % |

\* Near-zero because most 2026 ring annotations were seeded from the detection output and
only rings 6–8 were manually corrected by the annotator.

The algorithm never hard-fails (`success: false`) on any image, but has two distinct
failure modes that are severe on the new batch: **grey-zone ring detection** and
**arrow false positives**.

---

## Ring detection

### What works

Rings 0–5 (gold + red + blue zones) and ring 9 (outermost) are reliably accurate across
both batches:

| Ring | Zone | Mean error | Max error | Bad (> 15 px) |
|---|---|---|---|---|
| 0 (X)  | gold inner  | 1.4 px | 4.1 px | 0/25 |
| 1 (10) | gold outer  | 0.7 px | 3.5 px | 0/25 |
| 2 (9)  | red inner   | 0.7 px | 2.1 px | 0/25 |
| 3 (8)  | red outer   | 1.0 px | 2.2 px | 0/25 |
| 4 (7)  | blue inner  | 1.1 px | 3.2 px | 0/25 |
| 5 (6)  | blue outer  | 1.6 px | 4.4 px | 0/25 |
| 9 (2)  | white outer | 5.2 px | 9.5 px | 0/25 |

The yellow/red/blue colour-blob bootstrap gives a stable centre and scale, and the
colour-guided radial scan resolves gold/red/blue boundaries with < 6 px accuracy on
all 25 images.

### Failure: grey-zone rings 6–8 (5♦ / 4 / 3)

| Ring | Zone | Mean error | Max error | Bad (> 15 px) |
|---|---|---|---|---|
| 6 (5♦) | black inner | 29.5 px | 87.7 px | **12/25** |
| 7 (4)  | black outer | 46.2 px | 167.8 px | **11/25** |
| 8 (3)  | white inner | 31.8 px | 98.3 px | **12/25** |

All 12 bad images are from the 2026-03-19 evening batch. The 2019 batch has < 7 px on
these rings. 3 of 15 new-batch images also succeed (20260319_211823, 20260319_211830,
20260319_220656) — all three are images where the detected grey zone appeared visually
correct to the annotator and was left unchanged.

**Pipeline step responsible:** ring[7] is a directly-detected ring (colour-guided scan,
one of the 5 odd-indexed rings). Rings 6 and 8 are *interpolated* between ring[5]–ring[7]
and ring[7]–ring[9] respectively. Any error in ring[7] propagates into both neighbours.

**Root cause hypothesis:** The grey/black scoring zone has low chromatic saturation and
is detected via luminance transitions rather than hue. In the 2026 evening images, uneven
artificial lighting and a hay-bale backstop behind the target create luminance gradients
that mimic the grey-zone boundary. The adaptive re-centring that works well for yellow,
red and blue has no equivalent for grey, so the detector latches onto the wrong transition.

---

## Arrow detection

### 2019 batch — acceptable

| Metric | Value |
|---|---|
| Annotated | 65 |
| Detected  | 65 |
| TP | 50 |
| FN | 15 |
| FP | 15 |
| Recall | 77 % |
| Precision | 77 % |

Missed arrows are mostly in crowded clusters where two shafts are nearly collinear and the
deduplication step (`removeMidshaftDuplicates`) suppresses one of them.

### 2026-03-19 batch — poor precision

| Metric | Value |
|---|---|
| Annotated | 81 |
| Detected  | 178 |
| TP | 37 |
| FN | 44 |
| FP | 141 |
| Recall | 46 % |
| Precision | 21 % |

Per-image FP counts range from 0 to 25. The worst cases:

| Image | Ann | Det | FP |
|---|---|---|---|
| 20260319_205957.jpg | 6 | 27 | 25 |
| 20260319_214706.jpg | 5 | 25 | 21 |
| 20260319_213753.jpg | 6 | 15 | 14 |
| 20260319_215628.jpg | 6 | 14 | 12 |
| 20260319_212843.jpg | 6 | 12 | 11 |

**Pipeline step responsible:** The shaft mask (Hough segment extraction, stage D1) is
generating far too many raw segments. The subsequent rejection filters (dark-stripe
verification D4, deduplication D5–D6) are not removing them.

**Root cause hypothesis:** Outdoor evening images with a hay-bale backstop produce many
fine linear textures (straw fibres, shadow edges) that pass the dark-stripe check and
length filter. Arrow shaft segments are not sufficiently distinct from background line
features in these conditions.

---

## Per-image detail

| Image | Worst ring (px) | Ring status | Arrows TP/Ann | FP |
|---|---|---|---|---|
| 20190321_211008.jpg | 4.7 | OK | 5/9 | 2 |
| 20190321_212022.jpg | 3.3 | OK | 7/9 | 4 |
| 20190321_212956.jpg | 0.8 | OK | 3/6 | 2 |
| 20190325_193217.jpg | 3.6 | OK | 6/6 | 0 |
| 20190325_193820.jpg | 5.3 | OK | 5/6 | 2 |
| 20190325_195129.jpg | 5.0 | OK | 5/6 | 1 |
| 20190325_195801.jpg | 5.0 | OK | 6/6 | 0 |
| 20190325_201217.jpg | 6.3 | OK | 4/5 | 2 |
| 20190325_202607.jpg | 4.2 | OK | 4/6 | 1 |
| 20190325_204137.jpg | 5.0 | OK | 5/6 | 1 |
| 20260319_205957.jpg | 163.9 | **rings 6–8 fail** | 2/6 | **25** |
| 20260319_210000.jpg | 84.2 | **rings 6–8 fail** | 5/6 | 6 |
| 20260319_210902.jpg | 93.6 | **rings 6–8 fail** | 2/6 | **16** |
| 20260319_211823.jpg | 7.1 | OK | 2/6 | 7 |
| 20260319_211830.jpg | 9.5 | OK | 2/3 | 6 |
| 20260319_212836.jpg | 50.9 | **rings 6–8 fail** | 4/6 | 0 |
| 20260319_212838.jpg | 50.2 | **rings 6–8 fail** | 0/5 | 1 |
| 20260319_212843.jpg | 149.2 | **rings 6–8 fail** | 1/6 | **11** |
| 20260319_213753.jpg | 100.3 | **rings 6–8 fail** | 1/6 | **14** |
| 20260319_213758.jpg | 32.7 | **rings 6–8 fail** | 3/5 | 0 |
| 20260319_214706.jpg | 89.7 | **rings 6–8 fail** | 4/5 | **21** |
| 20260319_215628.jpg | 165.0 | **rings 6–8 fail** | 2/6 | **12** |
| 20260319_215640.jpg | 63.1 | **rings 6–8 fail** | 4/6 | 4 |
| 20260319_220650.jpg | 167.8 | **rings 6–8 fail** | 2/3 | **13** |
| 20260319_220656.jpg | 5.8 | OK | 3/6 | 5 |

---

## Pipeline step failure map

| Step | What it does | Failure mode observed |
|---|---|---|
| Pretreatment | Gaussian blur + erode/dilate | None observed |
| Colour blob detection | HSV yellow/red/blue blobs → centre + scale | None — 25/25 succeed |
| Bootstrap scale | Centre + ring-width `w` estimate | Stable across both batches |
| Boundary scan | 180 rays, median-filtered polygon | Boundary detected on all images |
| Colour-guided ring scan | 32 rays, zone classification + luminance for grey | **Fails on grey zone (ring[7]) in evening outdoor images** |
| Spline interpolation | Rings 0,2,4,6,8 from lerp of neighbours | **Cascades ring[7] error into rings[6] and [8]** |
| Arrow shaft mask | Hough segments on dark-pixel mask | **Over-generates in hay-bale outdoor backgrounds** |
| Arrow filters D3–D6 | Length, dark stripe, dedup | Insufficient for outdoor straw texture |

---

## Suggested fixes

### Rings 6–8

**Fix R1 — Add grey as a 4th colour blob.**
Extend `detectColorBlob` to detect the grey zone (HSV S < 0.2, V in 0.35–0.65) as a
fourth anchor. This gives a direct estimate of the ring[7] radius rather than relying
solely on luminance transitions. The grey blob centroid radius can be compared against
the expected WA ratio (grey outer ≈ 7.2w) to sanity-check.

**Fix R2 — Ratio-based sanity clamp for ring[7].**
After detecting ring[7], compute `r7 / r5`. The WA standard ratio is ≈ 8/6 = 1.333
(ring[7] ≈ 8w, ring[5] ≈ 6w). If the measured ratio falls outside [1.05, 1.65], snap
ring[7] to the predicted position `r5 × 1.333`. This catches luminance mistracking
without requiring a better detector.

**Fix R3 — Per-zone adaptive luminance calibration.**
The grey zone is currently detected with a fixed luminance threshold. Measure the actual
luminance of sampled grey-zone pixels (using the already-computed `sampleZoneColours`)
and adapt the threshold per image, similar to how hue ranges are adaptively re-centred
for yellow/red/blue.

**Fix R4 — Use ring[9] as an outer anchor for ring[7].**
Ring[9] (outermost) is reliably detected (< 10 px on all images). The expected ratio
`r9 / r7 ≈ 1.39` could constrain ring[7] from above, narrowing the search range for
the grey-zone scan.

### Arrow detection

**Design priority: precision over recall.** Missing an arrow is acceptable; a spurious
detection is not. All fixes below bias toward fewer, more confident detections.

**Fix A1 — Discard any tip or nock outside the paper boundary.**
Any detected arrow whose tip or nock lies outside `paperBoundary` is certainly a false
positive. A single `ptInPoly` call per candidate — zero additional computation. This alone
would eliminate most hay-bale detections.

**Fix A2 — Raise the minimum shaft length dynamically.**
Arrow shaft length in pixels scales with `w` (ring-width, already available). A minimum
of `0.3 × w` (≈ half the innermost ring radius) would discard most short straw-fibre
segments while keeping real shafts that cross at least one full zone.

**Fix A3 — Tighten the Hough threshold under high candidate counts.**
If raw segment count (D1) exceeds a reasonable ceiling (e.g. 3× the expected max arrows),
progressively raise the Hough vote threshold until candidates fall below it. Avoids
committing to a fixed threshold that is too loose for outdoor backgrounds.

**Fix A4 — Restrict shaft mask to the target area.**
Build a mask from pixels that lie within the paper boundary and inside the outermost ring.
Run the Hough transform only on that region. Straw, hay, and shadow edges outside the
target disk never enter the pipeline.
