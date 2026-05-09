# Improvement Failure Log

Documents attempted algorithm improvements that caused regressions or were reverted.
Each entry explains what was tried, what failed, and why.

---

## Erosion of the dark-pixel shaft mask (2026-03-25)

**Attempted fix:** Erode the `buildRelativeDarkMask` output by 1 pixel radius before
passing it to the Hough transform, to remove single-pixel straw-fibre segments that
produce false positive arrows.

**Implementation:** Added `erodeMask(mask, width, height, radius=1)` that replaces each
pixel with the minimum of its neighbourhood. Applied once to `shaftMask` in `findArrows`.

**Failure:** `20190321_212022.jpg` dropped from 11 to 4 detected arrows (7 required by
the ground truth assertion); `20190325_193820.jpg` dropped from 7 to 3 (4 required).

**Root cause:** Arrow shafts are only 1–2 px wide at the image resolution used (max
1200 px wide). An erosion radius of 1 px removes the shaft centre entirely, wiping both
the real shaft and the noise fibres. There is no erosion radius that is large enough to
remove straw fibres but small enough to preserve real shafts at this resolution.

**Resolution:** Reverted in full.

---

## A4 regression: 20190321_212022.jpg arrow detection (2026-03-25)

**Fix attempted:** A4 — restrict shaft mask to within outermost ring (`else continue;` instead
of `else thr = outerThr;` in `buildRelativeDarkMask`).

**Purpose:** Eliminate false-positive arrows from hay-bale backstop (outside ring[9]) in
the 2026 evening batch.

**Regression:** `20190321_212022.jpg` moved from passing to failing (was one of the 7
previously-passing images).

**Root cause:** Two annotated arrows (`ann[7]` and `ann[8]`) have tips at radius ~170 px
from center with ring[9] at ~200 px. Their shafts extend predominantly outside ring[9] —
the shaft portion inside ring[9] is only ~31 px long, below the minimum segment length
of 50 px. With A4 excluding pixels outside ring[9], those shaft segments never accumulate
enough Hough votes to survive the length filter.

**Attempts to fix the regression without reverting A4:**
1. `else thr = 0.15;` — strict V threshold outside ring[9]. Arrow shaft (V ≈ 0.05–0.12)
   passes; straw/hay (V ≈ 0.20+) rejected in theory. In practice caused a new regression
   on `20190321_211008.jpg` (other dark features outside ring[9] at V < 0.15 generate
   FP segments).
2. `else if (r <= rWhite * 1.5) thr = 0.12; else continue;` — same outcome.
3. Dynamic minimum shaft length `max(30, round(ring9 × 0.18))`: helped `20260319_212836`
   (a 2026 image) but caused a new regression on `20190321_211008.jpg`. Net zero change.

**Resolution:** Kept A4 (`continue`) since it provides a net improvement of 6 tests
(18 → 12 failing), with 1 regression on `212022`.

**Possible future fix:** Pass a ring-radius limit to `houghSegments` so only peaks derived
from pixels within ring[9] are generated, but shaft pixels outside ring[9] still count
toward segment length. This preserves shaft length for near-boundary arrows without
introducing hay-bale Hough peaks.

---
