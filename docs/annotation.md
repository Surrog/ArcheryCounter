# Annotation Tool — Documentation

`scripts/annotate.ts` is a local web-based tool for annotating archery target photos.
It serves a single-page browser UI backed by a Node.js HTTP server and a PostgreSQL database.

Start it with:

```bash
npm run annotate
```

The server opens `http://localhost:3737` automatically (set `NO_BROWSER=1` to suppress).
Use `ANNOTATE_PORT=<port>` to change the port.

---

## Architecture overview

```
Browser (HTML/JS)
  ↕ fetch /api/*
Node.js HTTP server  (scripts/annotate.ts)
  ↕ pg Pool
PostgreSQL  (annotations table)
  ↕ npm run seed-from-parquet / dump-annotations
data/annotations.parquet  (portable snapshot)
```

### Database table — `annotations`

| Column | Type | Description |
|---|---|---|
| `filename` | TEXT PK | JPEG filename (basename only) |
| `paper_boundary` | JSONB | Human-annotated boundary polygon `[x,y][]` or null |
| `rings` | JSONB | Human-annotated spline rings `SplineRing[]` |
| `arrows` | JSONB | Human-annotated arrows `ArrowAnnotation[]` |
| `updated_at` | TIMESTAMPTZ | Last save timestamp |
| `detected_rings` | JSONB | Algorithm-detected rings (cached) |
| `detected_boundary` | JSONB | Algorithm-detected boundary (cached) |
| `detected_arrows` | JSONB | Algorithm-detected arrows (cached) |
| `algorithm_hash` | TEXT | SHA-256 (16 chars) of source files at detection time |

### Algorithm hash and cache invalidation

At startup the server hashes `src/targetDetection.ts` and `src/arrowDetection.ts`.
Any DB row whose stored `algorithm_hash` differs from the current hash is treated as **stale**:
its `detected_*` columns are out of date and will be recomputed the next time the image is selected.

---

## Server API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves the full annotation HTML page |
| GET | `/api/annotations` | Returns all human annotations as `{ [filename]: { paperBoundary, rings, arrows } }` |
| GET | `/api/stale-images` | Returns `{ stale: string[] }` — filenames whose detection hash is outdated |
| GET | `/api/image-status/:filename` | Returns `{ state: "ready" \| "stale" \| "new" }` for one file |
| GET | `/api/image/:filename` | Returns `CachedImageData` (see below); runs detection if stale/new |
| POST | `/api/save` | Saves human annotations for one or more images to DB |

### `CachedImageData` response shape

```typescript
{
  base64: string;          // JPEG data-URL, scaled to max 1200 px
  width:  number;          // pixel width of the scaled image
  height: number;          // pixel height of the scaled image
  detected: {
    rings:         SplineRing[];                                 // algorithm output
    paperBoundary: [number, number][] | null;                   // algorithm output
    arrows:        { tip: [number, number]; nock: [number, number] | null }[];
  };
}
```

### Image loading — fast vs slow path

When a user selects an image:

1. **Fast path** (`algorithm_hash` matches): the server loads the base64 via Jimp and reads
   `detected_*` from the DB. No algorithm runs. Typical response time: ~1–2 s.

2. **Slow path** (stale or new): the server runs the full detection pipeline synchronously
   (`findTarget` → `findArrows`), stores results in the DB, and responds.
   Typical response time: ~30–60 s (blocks the event loop — single-user tool).

The browser shows a `Computing rings… Xs` counter while waiting on the slow path.

---

## Browser UI — layout

```
┌─────────────────┬─────────────────────────────────────┐
│  Sidebar        │  Main canvas                        │
│  ─ controls     │  (image + SVG overlay)              │
│  ─ view toggle  │                                     │
│  ─ checkboxes   │                                     │
│  ─ score picker │                                     │
│  ─ image list   │                                     │
│  ─ data panel   │                                     │
└─────────────────┴─────────────────────────────────────┘
```

---

## Step-by-step: selecting an image

1. On load the browser fetches `/api/annotations` to populate the in-memory store, then
   `/api/stale-images` to mark which files need recomputation.
2. The image list is rendered in the sidebar. Stale images appear in grey italic with a `·` suffix.
3. Clicking an image calls `selectImage(idx)`:
   - If already in `imageDataCache`, renders immediately.
   - Otherwise shows a loading indicator, fetches `/api/image-status/` to confirm stale/ready,
     starts an elapsed-seconds counter if computing, then fetches `/api/image/`.
4. On response, `imageDataCache[filename]` is populated and `render()` is called.

---

## Step-by-step: view modes

Press **Tab** or click the **Annotated / Generated** toggle to switch.

| Mode | Rings shown | Boundary shown | Arrows shown | Editable |
|---|---|---|---|---|
| Annotated | `ann.rings` (human) | `ann.paperBoundary` (human) | `ann.arrows` (human, orange) | Yes |
| Generated | `data.detected.rings` (algorithm) | `data.detected.paperBoundary` (algorithm) | `data.detected.arrows` (algorithm, cyan) | No |

A blue badge `read-only · algorithm output` appears under the toggle in Generated mode.
All drag handles and add-arrow mode are disabled in Generated mode.

---

## Step-by-step: annotating rings

Rings are stored as `SplineRing[]` — each ring is an array of 8 Catmull-Rom control points
that the browser smooths into a closed curve with 120 sampled points.

Ring index 0 = innermost (bullseye, gold), index 9 = outermost (white).
Colour mapping: 0–1 gold, 2–3 red, 4–5 blue, 6–7 grey, 8–9 white.

**Drag a control point** to reshape the ring.
Control-point handles are small filled circles; index 0 is labelled with the ring number.

Enable/disable the ring overlay with the **Rings** checkbox.
Enable/disable handles with the **Handles** checkbox.

---

## Step-by-step: annotating the boundary

The paper boundary is a polygon with 4–N vertices stored as `[number, number][]`.

- **Drag a vertex** (large green circle) to move it.
- **Ctrl + click** anywhere on the canvas to insert a new vertex on the nearest edge.
- **Shift + click** a vertex to remove it (minimum 3 vertices enforced).

Enable/disable the boundary overlay with the **Boundary** checkbox.

---

## Step-by-step: annotating arrows

Each arrow is:

```typescript
interface ArrowAnnotation {
  tip:   [number, number];       // impact point (scoring location)
  nock:  [number, number];       // rear of shaft
  score: number | 'X' | null;   // 1–10, 'X' (inner gold), 0 (miss), null (unscored)
}
```

### Adding an arrow

1. Press **A** or click **Add arrow** — mode enters `place-tip` (cursor becomes crosshair).
2. Click the canvas at the tip (impact point). Mode advances to `place-nock`.
   The tip appears as a dim orange dot.
3. Click at the nock (rear of shaft). Mode advances to `score-input`.
4. The score picker appears. Choose a score:
   - Click a button: **X**, **10–1**, **M** (miss = 0), **?** (null/unscored).
   - Or type: `x`/`X` → X, `m`/`M` → miss, `1`–`9` → that score.
   - **Escape** commits the arrow with `score: null`.
5. The arrow is added to `ann.arrows`, mode returns to `idle`.

Press **Escape** during `place-tip` or `place-nock` to cancel the whole arrow.
Press **A** again during `place-tip` to cancel and return to idle.

### Editing an arrow

- **Drag** the tip handle (red, r=4) or nock handle (gold, r=6) to reposition.
- **Shift + click** either handle to remove the arrow.
- Click the **score cell** in the data panel to reopen the score picker for that arrow.

### Arrow rendering

| Property | Visual |
|---|---|
| Shaft | Dashed line, orange (#FF8C00); grey if miss |
| Tip handle | Red (#FF4500) circle, dashed stroke if score is null |
| Nock handle | Gold (#FFD700) circle |
| Label | Score value near midpoint (white text); `M` for miss, `?` for null |
| Generated mode | Cyan (#00CFCF) line + circles, labelled by index |

---

## Step-by-step: saving

Click **Save** (or press nothing — there is no auto-save).
The browser POSTs all modified annotations to `/api/save`. The server upserts each row
into the DB, updating `paper_boundary`, `rings`, `arrows`, and `updated_at`.
Modified images lose their orange dot in the image list.

---

## Step-by-step: filtering the image list

Three filter buttons above the image list:

| Button | Shows |
|---|---|
| All | Every image |
| Annotated | Images with at least one arrow, ring, or boundary |
| Unannotated | Images with no annotation data |

An image is considered annotated if `ann.arrows.length > 0`, `ann.rings.length > 0`,
or `ann.paperBoundary != null`.

---

## Step-by-step: resetting

- **Reset image** — discards all annotation for the current image, reseeds from detected data,
  clears modified state.
- **Reset all** — discards all annotations in memory (requires confirmation).

Neither button saves to the DB automatically. Use **Save** to persist.

---

## Data panel

The bottom of the sidebar shows a summary of the current annotation:

- Boundary vertex count.
- Per-ring: index, control-point count, centroid (cx, cy).
- Per-arrow: index, tip coords, nock coords, score (clickable to re-score).

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| Tab | Toggle Annotated / Generated view |
| A | Toggle add-arrow mode (idle ↔ place-tip) |
| Escape | Cancel arrow placement / close score picker |
| 1–9 | Set score (during score-input mode) |
| x / X | Set score to X (during score-input mode) |
| m / M | Set score to 0 / miss (during score-input mode) |

---

## Data flow: parquet snapshot

The DB is the live store. The parquet file (`data/annotations.parquet`) is a portable snapshot:

- **`npm run dump-annotations`** — exports all rows (filename, paper_boundary, rings, arrows) to parquet.
- **`npm run seed-from-parquet`** — upserts parquet rows into the DB, restoring
  `paper_boundary`, `rings`, and `arrows` without touching `detected_*` or `algorithm_hash`.

---

## Planned improvements

<!-- Add future work here -->
