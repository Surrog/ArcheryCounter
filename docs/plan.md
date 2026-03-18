# ArcheryCounter — Implementation Plan

See `docs/research.md` for rationale behind every decision.

---

## Status (as of 2026-03-18)

Phases 1–7 are **complete and passing** (10/10 targetDetection + 10/10 groundTruth tests).

| Phase | Description | Status |
|---|---|---|
| P1 | Boundary detection (ray-cast + convex hull polygon) | ✅ |
| P2 | Per-image colour calibration (von Kries white-balance) | ✅ |
| P3 | Colour-guided radial ring detection (32 rays, 10 boundaries/ray) | ✅ |
| P4 | White ring / outer boundary detection (black-line scan + regression) | ✅ |
| P5 | Annotation tool migration to spline control points | ✅ |
| P6 | Algorithm output migrated to `SplineRing[]` | ✅ |
| P7 | Ground-truth test suite (PostgreSQL annotations, 10/10 images) | ✅ |
| P8 | Scoring pipeline | ⬜ deferred — needs arrow detection |
| P9 | Arrow detection | ⬜ in research |

---

## Pipeline (current)

```
Photo
  → [P1] Boundary polygon (4–8 vertices)
  → [P2] ColourCalibration (per-image HSV references per zone)
  → [P3] Colour-guided ring detection (32 rays × 10 boundaries)
      ↳ monotonicity enforced: forward pass on transitionDist[] + final pass on result[]
  → [P4] White-ring closure (black-line scan → regression → fallback)
  → [P6] SplineRing[10] (K=12 Catmull-Rom control points per ring)
Output: { rings, paperBoundary, calibration, ringPoints, rayDebug }
```

---

## Phase 8 — Scoring pipeline (deferred)

*Do not implement until Phase 9 (arrow detection) is underway.*

- [ ] **P8-T1** `classifyColourZone(hsv, cal): ColourZone | null`
- [ ] **P8-T2** `samplePatchZone(rgba, width, height, pt, radius, cal): ColourZone | null` — modal zone of annular patch, excludes hay pixels
- [ ] **P8-T3** `disambiguateScore(zone, pt, innerRing, outerRing): number` — distance to inner/outer spline → exact score
- [ ] **P8-T4** `isXRing(pt, centre, goldInnerSpline): boolean` — within inner 40% of gold zone radius
- [ ] **P8-T5** `scoreArrow(rgba, width, height, arrowTip, result): number | 'X' | 0`
- [ ] **P8-T6** Wire into `ArcheryCounter.processImage` once arrow tips are available

---

## Phase 9 — Arrow detection

See `docs/research.md §7` for the full research and approach comparison.

### Recommended implementation order

- [ ] **P9-T1** Nock detector: find small (5–20 px diameter), saturated, near-circular blobs in front of the target face. Use the calibration to exclude target-zone colors when possible; rely on blob shape and size otherwise.
- [ ] **P9-T2** Shaft direction prior: cast rays from each nock candidate inward toward the target center; the ray direction that minimises luminance (dark shaft) gives the shaft angle.
- [ ] **P9-T3** Impact point: trace the shaft ray until it intersects the target paper (luminance jump to target background color); the last dark pixel before the jump is the entry point.
- [ ] **P9-T4** Multi-arrow deduplication: merge nock candidates within 15 px; require one impact point per nock.
- [ ] **P9-T5** Arrow-hole fallback: if no shaft lines are found, detect small (~5–15 px) dark/warm circular regions (exposed hay) within the target boundary. Use as fallback impact positions.
- [ ] **P9-T6** Collect images with arrows in place (and post-pull holes) to build a detection test dataset.
- [ ] **P9-T7** Wire into Phase 8 scoring once impact points are reliable.

---

## Key design decisions (permanent)

- Ring index 0 = innermost (bullseye), index 9 = outermost. Score 10 = bullseye, 1 = outermost, 0 = miss.
- `BOOTSTRAP_SCALE = 2`: pretreatment runs on 2× downsampled image; centroids/radii scaled back.
- `N_BOUNDARY = 180`: boundary scan uses 180 rays; ring detection uses 32 rays.
- Monotonicity enforcement in `detectRingDistancesOnRay`: forward pass on `transitionDist[]` before result commit, then final pass on full `result[0..9]`. Both nullify violations (treated as missing, filled by interpolation on neighbouring rays).
