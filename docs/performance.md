# Detection Performance Report

Generated from 25 annotated images (10 from 2019-03 batch, 15 from 2026-03-19 batch).
Benchmark script: `scripts/benchmark.ts` — reads `annotations` vs `generated` tables.
Ring metric: symmetric mean boundary distance (px). Arrow match threshold: 40 px.

Ground-truth test status: **13/25 passing** (as of 2026-03-25, after R2 + A4 fixes).

---

## Ring detection

### Summary

| Batch | Rings 0–5 | Rings 6–8 | Ring 9 |
|---|---|---|---|
| 2019 (10 imgs) | < 7 px all images | < 7 px all images | < 7 px |
| 2026-03-19 (15 imgs) | < 2 px (seeded from detection) | **69 px mean** / 12 of 15 fail | 6.0 px |

Rings 0–5 (gold + red + blue) and ring 9 are reliable across both batches.
Rings 6–8 (black + white inner) fail on 12/15 evening images.

**Root cause:** Ring[7] (black→white transition) is detected via luminance gradients. Under
uneven evening artificial lighting with hay-bale backstop, the luminance gradient outside
the target mimics the black→white boundary.

**Fix R2 (implemented 2026-03-25):** After per-ray collection, if `r7/r5 < 1.05` or
`> 1.65` (expected ≈ 8/6 = 1.333), or ring[7] has < 3 detected points, ring[7] is rebuilt
from ring[5] points scaled outward by 8/6. Fixed 6 of the 12 failing images.

### Remaining ring fixes to explore

**Fix R1 — Add grey as a 4th colour blob.**
Extend `detectColorBlob` to detect the grey zone (HSV S < 0.2, V in 0.35–0.65) as a
fourth anchor, giving a direct estimate of ring[7] radius. Would help the 3 images whose
ring center is off by ~200px (210902, 213753, 214706) — these likely have a bootstrap
center error rather than a ring[7] radius error.

**Fix R3 — Per-zone adaptive luminance calibration.**
The grey zone is detected with a fixed luminance threshold. Adapt it using the actual
luminance of sampled grey-zone pixels (already measured by `sampleZoneColours`) per image,
similar to the adaptive hue re-centering for yellow/red/blue.

**Fix R4 — Use ring[9] as outer anchor for ring[7].**
Ring[9] is reliable (< 10 px all images). Expected `r9/r7 ≈ 10/8 = 1.25` could constrain
the ring[7] search range from above.

---

## Arrow detection

### Summary

| Batch | Annotated | TP | Recall | Precision |
|---|---|---|---|---|
| 2019 (10 imgs) | 65 | 50 | 77 % | 77 % |
| 2026-03-19 (15 imgs) | 81 | 37 | 46 % | 21 % |

**Root cause (FP):** Hough segments from straw/hay fibres and shadow edges on the hay-bale
backstop pass the dark-stripe check and length filter. Shaft segments are not
distinguishable from outdoor straw texture at pixel level.

**Fix A4 (implemented 2026-03-25):** Shaft mask excludes pixels outside ring[9] (`continue`
in `buildRelativeDarkMask`). Eliminates hay-bale pixels from Hough input. Fixed several 2026
images but caused 1 regression (`20190321_212022`: two arrows with tip near ring[9] edge have
only ~31px shaft inside ring[9], below the 50px minimum). See `docs/improvement_failure.md`.

### Remaining arrow fixes to explore

**Fix A1 — Already implemented.** Tips outside paper boundary are rejected in `filterSegments`.
Tips outside ring[9] are also filtered (lines 837–840 in `arrowDetection.ts`).

**Fix A2 — Dynamic minimum shaft length.**
Arrow shafts near the ring[9] edge have < 50px of shaft inside the scoring area after A4.
Attempted: `max(30, ring9 × 0.18)`. Caused regression on `20190321_211008`. See
`improvement_failure.md`. Possible fix: restrict Hough *voting* to within ring[9] while
preserving shaft pixels outside it for length counting.

**Fix A3 (dropped) — Shaft orientation toward target centre.**
Arrows don't reliably point at the centre (archers vary stance, shafts flex on impact).
