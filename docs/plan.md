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
| P9 | Arrow detection algorithm | ✅ |
| P10 | Scoring pipeline | ⬜ next |

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

## Key design decisions

- Ring index 0 = innermost (bullseye), index 9 = outermost. Score 10 = bullseye, 1 = outermost, 0 = miss.
- `BOOTSTRAP_SCALE = 2`: pretreatment runs on 2× downsampled image; centroids/radii scaled back.
- `N_BOUNDARY = 180`, `N_RINGS = 32`.
- Monotonicity enforcement in `detectRingDistancesOnRay`: forward pass on `transitionDist[]` + final pass on `result[0..9]`. Both nullify violations; missing rings filled by interpolation on neighbouring rays.
- Arrow detection precision-first: missing arrows acceptable, false positives are not.
