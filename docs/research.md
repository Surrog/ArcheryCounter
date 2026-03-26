# ArcheryCounter — Research Notes

---

## §1 Why free-form splines (not ellipses or quads)

An archery target face is non-planar: gravity sag, arrow damage, edge curl, and humidity deform it. Its image is therefore **not** a projective transform of the canonical flat target.

- **Ellipse fit fails** — inner rings (bullseye) are most deformed; true conics only if the surface is flat.
- **4-corner quadrilateral fails** — curved edges between pins cannot be represented by four straight segments.

**Resolution:** Catmull-Rom SplineRings (K=12 control points) for ring boundaries; 4–8 vertex polygon for the target boundary.

---

## §2 Representations

| Representation | Use |
|---|---|
| Free-form closed spline (K=12 Catmull-Rom) | Ring output + annotation ground truth |
| 4–8 vertex polygon | Target boundary |
| Radial profile (N×10 distances) | Internal detection only — too large to annotate manually |

---

## §3 Detection algorithm (`targetDetection.ts`)

1. **Pretreatment** — Gaussian blur (15×15, σ=1.5) + erode×1 + dilate×3 on 2× downsampled image.
2. **Colour blob detection** — Two-pass HSV filtering (wide range → adaptive re-centering around measured median hue) for yellow, red, blue zones → centre + ring-width `w`.
3. **Boundary scan** — 180 rays; circular median filter (±5 rays); fit 4–8 vertex polygon.
4. **Colour calibration** — `sampleZoneColours` (8 rays × 2 samples/zone); von Kries white-balance correction.
5. **Colour-guided ring detection** — 32 rays; 5-point mode-smooth zone classifications; detect 4 colour-zone transitions (gold→red, red→blue, blue→black, black→white) per ray with MIN_STREAK=10, MIN_ZONE_WIDTH=0.4w.
6. **R2 ratio clamp** — After per-ray collection, if `r7/r5 < 1.05` or `> 1.65` (or ring[7] < 3 pts), rebuild ring[7] from ring[5] scaled by 8/6 ≈ 1.333.
7. **White-ring extrapolation** — OLS linear regression through detected colour transitions; clamped to paper boundary distance.
8. **Spline construction** — Detected rings [1,3,5,7,9] → SplineRings; [0,2,4,6,8] filled by `lerpSpline` between adjacent detected zone boundaries.

**HSV convention:** H 0–360°, S/V 0–1. Wide initial ranges adaptively re-centred per image.

---

## §4 Arrow detection (`arrowDetection.ts`)

Zone-adaptive relative-dark-pixel Hough on shaft mask (within paper boundary and within outermost ring). Pipeline: raw Hough → collinear merge → filterSegments (tip-in-boundary, anti-ring) → verifyDarkStripe → second merge → length ≥ 50px → deduplicateTips → removeMidshaftDuplicates → matchVanes.

**Design priority: precision over recall.** Missing an arrow is acceptable; a spurious detection is not.

Known failure mode: straw/hay fibres inside ring[9] in evening outdoor images pass the dark-stripe check. See `docs/performance.md` and `docs/improvement_failure.md` for details and attempted fixes.

---

## §5 Scoring pipeline (`src/scoring.ts`)

Score = 10 − i, where i is the index of the innermost ring containing the tip (rings[0..9] walked inward-first via `pointInClosedSpline`). Outside all rings = 0 (miss).

**X-ring:** inside ring[0] AND `dist_from_centre < 0.4 × splineRadius(rings[1])`.

**Colour cross-check:** `samplePatchZone` samples an annular patch (r=4..12 px) around the tip, takes the modal `ZoneName` (excludes S < 0.15 hay pixels), and compares to the geometric score's expected zone. On disagreement, `ScoredArrow.lowConfidence = true` is set — the geometric result is never silently overridden.

| Zone | Expected scores |
|---|---|
| gold | 10, 9 (X) |
| red | 8, 7 |
| blue | 6, 5 |
| black | 4, 3 |
| white | 2, 1 |

Output type: `ScoredArrow { tip, nock, score: number | 'X', lowConfidence? }`. `ProcessImageResult.arrows` is `ScoredArrow[]`.

---

## §6 Neural network approach (future)

Root cause of current failures: colour/luminance rules don't generalise across lighting. A trained network could learn lighting-invariant features.

| Property | Current pipeline | Neural network |
|---|---|---|
| Training data | None (rule-based) | 200–500+ labelled images |
| Generalisation | Brittle to new lighting | Improves with diverse data |
| Mobile runtime | Pure TS, no extra dep | Requires TFLite / ONNX runtime (~4 MB INT8) |
| Inference time | ~500 ms JS | ~30–150 ms on device |

**Architecture:** Two-head MobileNetV2 (pretrained ImageNet, freeze early layers).
- Ring head: global avg pool → FC → `(cx, cy)` + 9 radii, all normalised by `max(w, h)`.
- Arrow head: 1×1 Conv + 8× upsample → 80×80 Gaussian heatmap of tip locations.

**Data:** minimum ~200 images; augment with colour jitter ±40%, rotation ±15°, perspective warp, random crop/scale. Semi-supervised expansion: pseudo-label unlabelled photos with the current pipeline, keep only `success === true` with monotonically increasing radii (ratio < 1.5).

**Training:** PyTorch + AdamW, loss = L1(centre) + masked-L1(radii) + 0.1×monotonicity penalty + 0.5×BCE(heatmap). Export ONNX → INT8 TFLite + `.ort` for `onnxruntime-react-native`.

**Post-processing:** model radii → circular `SplineRing[]` (K=12); heatmap peaks via NMS (R=3, threshold=0.5) → tip list; scoring via existing `scoreArrow` from §5.

**Limitations:** rings predicted as circles (no deformation); 80×80 heatmap limits tip precision to ~4 px; needs 200+ images minimum.

---

## §7 Excluded approaches

| Approach | Reason |
|---|---|
| Homography-based canonical scoring | Corrects perspective only; does not address surface deformation |
| Lens distortion correction | Algorithm works in image space; no rectified input assumed |
| Shaft direction prior (all shafts share vanishing point) | Per-image angle spread 14–111°; no reliable shared direction |
| Doublet filter (specular shaft signature) | Good refinement step but requires shaft direction first |
| Arrow-hole detection fallback | Stub reserved for second iteration |
| Erosion of shaft mask | Shafts are 1–2 px wide — erosion removes shafts and straw alike |
