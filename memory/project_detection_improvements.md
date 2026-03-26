---
name: Detection improvement progress
description: Current state of groundTruth test results and fixes applied to ring/arrow detection
type: project
---

As of 2026-03-25, groundTruth test results improved from 18 failing → 12 failing (13/25 passing).

**Fix R2 implemented** (`src/targetDetection.ts`, inside `findTarget()`):
Ratio-based sanity clamp for ring[7]. When `r7/r5` is outside [1.05, 1.65] OR transitionPoints[7]
has < 3 points, ring[7] is rebuilt from `transitionPoints[5]` scaled by 8/6 (≈ 1.333).
Using ring[5] (not ring[7]) as source preserves full 32-ray angular coverage.

**Fix A4 implemented** (`src/arrowDetection.ts`, in `buildRelativeDarkMask()`):
Pixels outside `rWhite` (ring[9] radius) now `continue` instead of using `outerThr = 0.30`.
Eliminates hay-bale backstop pixels from Hough input.

**Known A4 regression**: `20190321_212022.jpg` (was passing, now failing). Two arrows have
tips at ~170px from center with ring[9] at ~200px — the shaft inside ring[9] is ~31px, below
the 50px minimum. Three fix attempts all caused other regressions (documented in
`docs/improvement_failure.md`). Possible future fix: restrict Hough *voting* area to within
ring[9] while preserving shaft pixels outside it for length measurement.

**Remaining 12 failures breakdown**:
- Ring center errors (3 images: 210902, 213753, 214706) — bootstrap center detection fails
- Paper boundary errors (5 images: 212956, 211830, 212838, 215640, 220656) — boundary polygon off
- Arrow count failures (3 images: 212022 A4-regression, 211823 FP hay-bale inside ring[9], 212836 FN)
- No annotation in DB (1 image: 213758)

**Why:** Performance plan from `docs/performance.md`. R2 addresses grey-zone ring detection
failure under evening outdoor lighting. A4 addresses hay-bale FP arrows.

**How to apply:** Next improvements to try: Fix R1 (grey blob bootstrap) or Fix R3 (adaptive
grey luminance calibration) for the remaining ring-center failures; for arrow FPs in 211823,
investigate whether they originate inside ring[9] (hay bale visible through target holes).
