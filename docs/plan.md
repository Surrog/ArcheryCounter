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
| P8 | Arrow annotation in annotate tool | ⬜ next |
| P9 | Arrow detection (algorithm) | ⬜ after P8 |
| P10 | Scoring pipeline | ⬜ after P9 |

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

## Phase 8 — Arrow annotation in the annotate tool

Add the ability to annotate arrows in `scripts/annotate.ts`. Each arrow is two points: **tip** (impact point on the target face) and **nock** (rear end of the shaft). Both points are draggable. Arrows can be added and removed interactively.

### Data model

```typescript
interface ArrowAnnotation {
  tip:   [number, number];          // approximate shaft entry point (may be outside target boundary for misses)
  nock:  [number, number];          // rear of shaft (nock/vane end)
  score: number | 'X' | null;       // 0 (miss), 1–10, 'X' (inner gold), or null if not yet scored
}
// Per-image annotation becomes:
{ paperBoundary: [number,number][] | null, rings: SplineRing[], arrows: ArrowAnnotation[] }
// Note: score 0 = miss (arrow did not land on the scoring face). tip/nock are still recorded
// so the shaft can be rendered; the tip may lie outside the paper boundary.
```

### Tasks

- [x] **P8-T1 DB migration** — Add `arrows JSONB NOT NULL DEFAULT '[]'` column to the `annotations` table. Update the `CREATE TABLE IF NOT EXISTS` DDL. Run `ALTER TABLE annotations ADD COLUMN IF NOT EXISTS arrows JSONB NOT NULL DEFAULT '[]'` on startup before any reads.

- [x] **P8-T2 State model** — Add `arrows: ArrowAnnotation[]` to the in-memory annotation object. On load from `/api/annotations`, set `arrows = ann.arrows ?? []`. On `resetCurrent()` and `resetAll()`, initialise `arrows: []`.

- [x] **P8-T3 Add-arrow mode** — Introduce a mode variable `let addArrowMode: 'idle' | 'place-tip' | 'place-nock' = 'idle'` and a pending tip `let pendingTip: [number,number] | null = null`.
  - Add **"Add arrow"** button to the sidebar controls (below Reset).
  - Clicking the button sets `addArrowMode = 'place-tip'`.
  - First SVG click (not on a handle, not Ctrl/Shift): records tip, advances to `'place-nock'`.
  - Second SVG click: records nock, advances to `'score-input'` sub-state.
  - In `'score-input'`: a compact score picker appears in the toolbar — buttons `X 10 9 8 7 6 5 4 3 2 1 M` (M = miss, score 0). Clicking a button sets `score`, commits `{ tip, nock, score }` to `ann.arrows`, resets to `'idle'`. Typing `1`–`9`, `x`/`X` (→'X'), or `m`/`M` (→0, miss) also commits; `0` is not bound to avoid conflict with future shortcuts. Pressing **Escape** commits with `score: null`.
  - Pressing **Escape** during `'place-tip'` or `'place-nock'` cancels the arrow entirely and resets to `'idle'`.
  - Pressing **A** (when no input focused) toggles `'idle' ↔ 'place-tip'`.

- [x] **P8-T4 Arrow rendering** — In `render()`, for each arrow `{tip, nock}` at index `i`:
  - Dashed orange line from tip to nock (`stroke="#FF8C00"`, `stroke-width="2"`, `stroke-dasharray="8 4"`).
  - **Tip handle**: `r=8`, `fill="#FF4500"`, `data-handle="arrow_tip"`, `data-ai="${i}"`.
  - **Nock handle**: `r=6`, `fill="#FFD700"`, `data-handle="arrow_nock"`, `data-ai="${i}"`.
  - Label showing the score (`score === 0 ? 'M' : score === null ? '?' : String(score)`) in white 10px monospace near the midpoint. Arrows with `score === null` render the tip handle with a dashed stroke to indicate incomplete annotation. Arrows with `score === 0` (miss) render the shaft line in grey instead of orange.
  - While `addArrowMode === 'place-nock'`, render `pendingTip` as a dim orange dot (r=6, opacity=0.5).
  - Add **"Arrows"** checkbox to the toolbar. When unchecked, skip all arrow rendering.
  - SVG cursor is `crosshair` while in add-arrow mode; status text in the toolbar shows "Click to place tip" / "Click to place nock".

- [x] **P8-T5 Arrow drag handling** — Extend `attachSvgListeners()` to handle `data-handle === 'arrow_tip'` and `'arrow_nock'`:
  - `drag = { type: 'arrow_tip'|'arrow_nock', ai }`.
  - `mousemove` updates `ann.arrows[drag.ai].tip` or `.nock`.
  - Skip mousedown when `addArrowMode !== 'idle'`.

- [x] **P8-T6 Arrow removal** — Shift+click on any arrow handle removes that arrow: `ann.arrows.splice(ai, 1)`, `markModified`, `render()`.

- [x] **P8-T7 Save / load** —
  - `GET /api/annotations`: include `arrows: row.arrows` in each entry.
  - `POST /api/save`: add `arrows = EXCLUDED.arrows` to the upsert clause; pass `JSON.stringify(ann.arrows ?? [])` as `$4`. Seeding defaults to `[]`.

- [x] **P8-T8 Data panel** — Add an "Arrows" section showing count and one row per arrow with tip coordinates and score. Clicking a score cell opens the score picker inline for correction.

---

## Phase 9 — Arrow detection (algorithm)

See `docs/research.md §7` for the full research and approach comparison.

*Implement after P8 so a ground-truth dataset can be collected first.*

- [ ] **P9-T1** Shaft direction prior: 1D Hough accumulator over line angles in the target region; all shafts share a common vanishing point direction.
- [ ] **P9-T2** Shaft line detection: Hough/LSD filtered to estimated direction ±10°; each segment with one endpoint on the target surface is a candidate shaft.
- [ ] **P9-T3** Impact point: the segment endpoint closest to the target centre is the entry point.
- [ ] **P9-T4** Multi-arrow deduplication: merge nearby shaft-line candidates; require one impact point per shaft.
- [ ] **P9-T5** Arrow-hole fallback: if no shaft lines are found, detect small (~5–15 px) dark/warm circular regions (exposed hay) within the target boundary.
- [ ] **P9-T6** Collect images with arrows in place (and post-pull holes) using the P8 annotation tool; build a detection test dataset.
- [ ] **P9-T7** Add arrow ground-truth tests to `src/__tests__/groundTruth.test.ts`: for each image with annotated arrows, run `findArrows(rgba, width, height, targetResult)` and assert:
  - Detected arrow count matches annotated count (exact).
  - Each annotated tip is within 15 px of the nearest detected tip (bijective matching — each annotation pairs with exactly one detection).
  - Each annotated nock is within 25 px of its matched detection's nock (looser tolerance: nock position is less critical for scoring).
  - For arrows where `score !== null`: `detectedScore === annotatedScore` for each matched pair. Arrows with `score: null` are skipped for the score assertion.
- [ ] **P9-T8** Wire into Phase P10 scoring once impact points are reliable.

---

## Phase 10 — Scoring pipeline

*Implement after P9 (arrow detection).*

- [ ] **P10-T1** `classifyColourZone(hsv, cal): ColourZone | null`
- [ ] **P10-T2** `samplePatchZone(rgba, width, height, pt, radius, cal): ColourZone | null` — modal zone of annular patch, excludes hay pixels
- [ ] **P10-T3** `disambiguateScore(zone, pt, innerRing, outerRing): number` — distance to inner/outer spline → exact score
- [ ] **P10-T4** `isXRing(pt, centre, goldInnerSpline): boolean` — within inner 40% of gold zone radius
- [ ] **P10-T5** `scoreArrow(rgba, width, height, arrowTip, result): number | 'X' | 0`
- [ ] **P10-T6** Wire into `ArcheryCounter.processImage` once arrow tips are available
- [ ] **P10-T7** Add scoring assertions to `src/__tests__/groundTruth.test.ts`: for each image with arrows that have non-null annotated scores, run the full scoring pipeline and assert `scoreArrow(...)` returns the annotated score exactly (scores are discrete — no tolerance; `score: 0` means miss and `scoreArrow` must return `0`). Images where any arrow has `score: null` are skipped.

---

## Key design decisions (permanent)

- Ring index 0 = innermost (bullseye), index 9 = outermost. Score 10 = bullseye, 1 = outermost, 0 = miss.
- `BOOTSTRAP_SCALE = 2`: pretreatment runs on 2× downsampled image; centroids/radii scaled back.
- `N_BOUNDARY = 180`: boundary scan uses 180 rays; ring detection uses 32 rays.
- Monotonicity enforcement in `detectRingDistancesOnRay`: forward pass on `transitionDist[]` before result commit, then final pass on full `result[0..9]`. Both nullify violations (treated as missing, filled by interpolation on neighbouring rays).
