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
PostgreSQL  (annotations + generated tables)
  ↕ npm run seed-from-parquet / dump-annotations
data/annotations.parquet  (portable snapshot)
```

### Database tables

**Target schema** (see AW-1 in `annotation_works.md` for migration plan):

#### `annotations` — human-authored data only

| Column | Type | Description |
|---|---|---|
| `filename` | TEXT PK | JPEG filename (basename only) |
| `paper_boundary` | JSONB | Human-annotated boundary polygon `[x,y][]` or null |
| `rings` | JSONB | Human-annotated spline rings `SplineRing[]` |
| `arrows` | JSONB | Human-annotated arrows `ArrowAnnotation[]` |
| `updated_at` | TIMESTAMPTZ | Last save timestamp |

#### `generated` — algorithm-detected data, keyed by source hash

| Column | Type | Description |
| `filename` | TEXT PK | JPEG filename |
| `algorithm_hash` | TEXT | SHA-256 (16 chars) of detection source files |
| `paper_boundary` | JSONB | Detected boundary polygon or null |
| `rings` | JSONB | Detected spline rings `SplineRing[]` |
| `arrows` | JSONB | Detected arrows `ArrowDetection[]` |
| `updated_at` | TIMESTAMPTZ | When detection ran |

Keeping the two tables separate means a save can never accidentally overwrite cached
detections, and a recompute can never overwrite a human annotation.

**Current schema** (pre-migration): all columns live in a single `annotations` table,
with `detected_rings`, `detected_boundary`, `detected_arrows`, and `algorithm_hash`
columns alongside the human annotation columns.

### Algorithm hash and cache invalidation

At startup the server hashes `src/targetDetection.ts`.
Any row in `generated` whose stored `algorithm_hash` differs from the current hash is
treated as **stale** and will be recomputed the next time the image is selected.

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
    rings:         SplineRing[];
    paperBoundary: [number, number][] | null;
    arrows:        { tip: [number, number]; score: number | 'X' | null }[];
  };
}
```

### Image loading — fast vs slow path

When a user selects an image:

1. **Fast path** (`algorithm_hash` matches and stored data is valid): the server loads the
   base64 via Jimp and reads detected values from the DB. No algorithm runs.
   Typical response time: ~1–2 s.
   If the stored data contains invalid values (null coordinates, corrupt JSON), the server
   falls back to the slow path and recomputes. Invalid data events are written to a log file
   (see AW-6 in `annotation_works.md`).

2. **Slow path** (stale, new, or invalid cache): the server runs the full detection pipeline
   synchronously (`findTarget`), stores results in the DB, and responds.
   Typical response time: ~30–60 s (blocks the event loop — single-user tool).
   The browser shows a `Computing rings… Xs` animated counter while waiting.

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
│  ─ data panel ▼ │                                     │  ← collapsible (AW-3)
└─────────────────┴─────────────────────────────────────┘
```

The data panel is collapsible so the image list remains fully visible on short screens (see AW-3).

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
**Alt + click** anywhere on the canvas to insert a new control point on the nearest ring segment.
**Shift + click** a control point to remove it (minimum 3 points enforced).
Control-point handles are small filled circles; index 0 is labelled with the ring number.

Enable/disable the ring overlay and its handles with the **Rings** checkbox.

---

## Step-by-step: annotating the boundary

The paper boundary is a polygon with 4–N vertices stored as `[number, number][]`.

- **Drag a vertex** (large green circle) to move it.
- **Ctrl + click** anywhere on the canvas to insert a new vertex on the nearest edge.
- **Shift + click** a vertex to remove it (minimum 3 vertices enforced).

Enable/disable the boundary overlay and its handles with the **Boundary** checkbox.

---

## Step-by-step: annotating arrows

Each arrow is:

```typescript
interface ArrowAnnotation {
  tip:   [number, number];     // impact point (scoring location)
  score: number | 'X' | null; // 1–10, 'X' (inner gold), 0 (miss), null (unscored)
}
```

### Adding an arrow

1. Press **A** or click **Add arrow** — mode enters `place-tip` (cursor becomes crosshair).
2. Click the canvas at the tip (impact point). Mode advances to `score-input`.
3. The score picker appears near the tip. Choose a score:
   - Click a button: **X**, **10–1**, **M** (miss = 0), **?** (null/unscored).
   - Or type: `x`/`X` → X, `m`/`M` → miss, `1`–`9` → that score.
   - **Escape** commits the arrow with `score: null`.
4. The arrow is added to `ann.arrows`, mode returns to `idle`.

When detected rings are available, the score picker pre-selects the ring the tip falls in.

Press **Escape** during `place-tip` to cancel the whole arrow.
Press **A** again during `place-tip` to cancel and return to idle.

### Editing an arrow

- **Drag** the tip handle to reposition.
- **Shift + click** the tip handle to remove the arrow.
- Click the **score cell** in the data panel to reopen the score picker for that arrow.

### Arrow rendering

| Property | Visual |
|---|---|
| Tip handle | Red (#FF4500) circle, dashed stroke if score is null |
| Label | Score value above tip (white text); `M` for miss, `?` for null |
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
- Per-arrow: index, tip coords, score (clickable to re-score).

The panel is collapsible via a toggle button at its header (see AW-3).

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| Tab | Toggle Annotated / Generated view |
| A | Toggle add-arrow mode (idle ↔ place-tip) |
| N | Save current annotations and advance to the next image |
| Escape | Cancel arrow placement / close score picker |
| 1–9 | Set score (during score-input mode) |
| x / X | Set score to X (during score-input mode) |
| m / M | Set score to 0 / miss (during score-input mode) |

---

## Data flow: parquet snapshot

The DB is the live store. The parquet file (`data/annotations.parquet`) is a portable snapshot:

- **`npm run dump-annotations`** — exports all rows (filename, paper_boundary, rings, arrows) to parquet.
- **`npm run seed-from-parquet`** — upserts parquet rows into the DB, restoring
  `paper_boundary`, `rings`, and `arrows` without touching detection data.
