# Annotation Tool — Work Plan

Atomic improvement tasks for `scripts/annotate.ts`. Each task is self-contained and
can be implemented and tested independently. Tasks are roughly ordered by dependency.

---

## AW-1 — Separate `annotations` and `generated` DB tables

**Goal:** Keep human annotations and algorithm output in distinct tables so a save can never
overwrite a cached detection and a recompute can never overwrite a human annotation.

### Schema changes

```sql
-- human data only (existing columns, drop detected_*)
ALTER TABLE annotations
  DROP COLUMN IF EXISTS detected_rings,
  DROP COLUMN IF EXISTS detected_boundary,
  DROP COLUMN IF EXISTS detected_arrows,
  DROP COLUMN IF EXISTS algorithm_hash;

-- new table for algorithm output
CREATE TABLE IF NOT EXISTS generated (
  filename       TEXT PRIMARY KEY,
  algorithm_hash TEXT NOT NULL,
  paper_boundary JSONB,
  rings          JSONB NOT NULL DEFAULT '[]',
  arrows         JSONB NOT NULL DEFAULT '[]',
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### Server changes

- `main()` startup: create `generated` table; drop the `ALTER TABLE` lines that add
  `detected_*` columns to `annotations`.
- Startup stale check: `SELECT filename, algorithm_hash FROM generated` instead of `annotations`.
- `/api/stale-images`: compare against `generated.algorithm_hash`.
- `/api/image-status/`: look up `generated` table.
- `/api/image/` fast path: `SELECT rings, paper_boundary, arrows FROM generated WHERE filename=$1`.
- `/api/image/` slow path INSERT/UPDATE: write to `generated`, not `annotations`.
- `/api/save` POST: write to `annotations` only (no change to `generated`).
- `dump-annotations`: reads `annotations` only (no change needed).
- `seed-from-parquet`: writes `annotations` only (no change needed).

### Validation helper

```typescript
function isValidDetected(data: { rings: SplineRing[]; paperBoundary: ... }): boolean {
  return data.rings.every(r => r.points?.every(p => p[0] != null && p[1] != null));
}
```
Used on fast path: if `isValidDetected` returns false, fall through to slow path and
delete the bad `generated` row before recomputing.

### Tests (`src/__tests__/scripts.test.ts`)

- After startup, both `annotations` and `generated` tables exist.
- `POST /api/save` updates `annotations` but leaves `generated` unchanged.
- After a slow-path image load, `generated` has a row with `algorithm_hash = currentHash`.
- After a second request for the same image, the fast path is taken (no recomputation log).

---

## AW-2 — Fast path validity check

**Goal:** If the `generated` row exists with the right hash but contains corrupt data
(null coordinates, empty rings where detection should have succeeded), treat it as stale
and recompute rather than serving broken data to the browser.

### Changes

In the fast path of `/api/image/`:

```typescript
const isValid = detectedRings.every(r =>
  r.points?.every(p => p[0] != null && p[1] != null)
);
if (!isValid) {
  // delete the stale row and fall through to slow path
  await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
  inDb.delete(filename);
  // ... slow path runs below
}
```

### Tests

- Manually insert a `generated` row with `rings = '[{"points":[[null,null]]}]'` and correct hash.
- `GET /api/image/<filename>` should trigger the slow path (check log output contains `[3/4] findTarget`).
- After the request the `generated` row should have valid ring data.

---

## AW-3 — Collapsible data panel

**Goal:** On short screens the data panel eats into the image list. A collapse toggle
keeps the panel out of the way without removing it entirely.

### Changes

**HTML:** Add a `▼ / ▶` toggle button to the data panel header.

```html
<div id="data-panel">
  <div id="data-panel-header">
    <h3>Current annotation</h3>
    <button id="btn-collapse-data" title="Collapse">▼</button>
  </div>
  <div id="data-table"></div>
</div>
```

**CSS:** Collapsed state hides `#data-table` with zero max-height and no padding.

```css
#data-panel.collapsed #data-table { display: none; }
#data-panel.collapsed { padding-bottom: 0; }
#btn-collapse-data { ... }
```

**JS:**
- Toggle `collapsed` class on `#data-panel`.
- Persist state: `localStorage.setItem('dataPanelCollapsed', '1')`.
- Restore on load: `if (localStorage.getItem('dataPanelCollapsed')) ...`.
- Button label flips between `▼` (expanded) and `▶` (collapsed).

### Tests (`src/__tests__/scripts.test.ts` — annotate server test)

- `GET /` HTML contains `id="btn-collapse-data"`.
- `id="data-table"` and `id="data-panel"` exist in the DOM.
- (Visual/manual): collapsing the panel allows the image list to extend to the bottom.

---

## AW-4 — Auto-deduce arrow score from detected rings

**Goal:** When the user places a tip and rings are available, pre-select the ring the tip
falls inside so they just confirm rather than hunting for the right score.

### Algorithm

Given tip coordinates `(tx, ty)` and the 10 detected (or annotated) rings ordered
innermost-first:

1. Sample each ring's spline into a polygon (reuse the browser-side `sampleClosedSpline`).
2. Walk rings outward (index 0 → 9): the first ring whose polygon contains `(tx, ty)` determines the score.
   - Ring 0 → score `10` (or `X` — display as `10*` with a visual hint).
   - Ring k → score `10 - k`.
3. If the tip is outside all rings but inside `paperBoundary` → score `1` (outermost).
4. If the tip is outside `paperBoundary` → score `0` (miss).

### Changes

**Browser JS** — new function:

```javascript
function deduceScore(tipX, tipY, rings, boundary) {
  for (let i = 0; i < rings.length; i++) {
    if (!rings[i].points || rings[i].points.length < 3) continue;
    const poly = sampleClosedSpline(rings[i].points, 64);
    if (ptInPoly(tipX, tipY, poly)) return 10 - i;
  }
  if (boundary && ptInPoly(tipX, tipY, boundary)) return 1;
  return 0;
}
```

In `selectImage` / `place-nock` → `score-input` transition:
- Compute `const suggested = deduceScore(pendingTip[0], pendingTip[1], rings, boundary)`.
- Highlight the suggested button in the score picker with a distinct style (e.g. blue border).
- The user can still click any other button to override.

**Score picker HTML/CSS:** add a `.suggested` class that draws a subtle ring around the
button without changing its colour.

### Tests

- Unit test `deduceScore` with mock ring polygons:
  - Point at centre → 10.
  - Point between ring 2 and ring 3 → 8.
  - Point outside all rings but inside boundary → 1.
  - Point outside boundary → 0.
- These tests run in Node.js (extract `deduceScore` to a shared utility or test via the
  browser test harness).

---

## AW-5 — Non-blocking background generation

**Goal:** The server starts immediately; stale/new images are computed in the background
without blocking the event loop or delaying startup. The browser UI reflects per-image
computation status in real time.

### Architecture

```
main()
  ├─ DB setup + validate existing data          (fast, <1 s)
  ├─ HTTP server.listen()                        (server is immediately reachable)
  └─ backgroundQueue.start()                     (processes images one at a time)
        ↓ for each stale/new image
        spawnSync / worker + ts-node → findTarget + findArrows
        ↓ on completion
        UPDATE generated SET ...
        SSE broadcast { filename, status: 'ready' }
```

### Server changes

**Queue manager:**

```typescript
class DetectionQueue {
  private queue: string[];       // filenames pending
  private running = false;

  constructor(filenames: string[]) { this.queue = [...filenames]; }

  start(onDone: (filename: string) => void) {
    const next = () => {
      const filename = this.queue.shift();
      if (!filename) { this.running = false; return; }
      this.running = true;
      setImmediate(async () => {
        await processImage(path.join(IMAGES_DIR, filename));
        // write to generated table
        onDone(filename);
        next();
      });
    };
    next();
  }

  status(filename: string): 'ready' | 'queued' | 'computing' { ... }
}
```

**SSE endpoint** `GET /api/events`:

```
Content-Type: text/event-stream
data: {"type":"status","filename":"foo.jpg","state":"computing"}
data: {"type":"status","filename":"foo.jpg","state":"ready"}
```

**New `/api/generation-status`** endpoint: returns a map of all filenames to
`"ready" | "queued" | "computing"` for the initial page load.

**Browser changes:**

- On load, fetch `/api/generation-status` and merge with `staleImages`.
- Subscribe to `new EventSource('/api/events')`.
- Per-image badge in the list: a small coloured dot:
  - Green = has human annotation.
  - Blue = generated and ready (no human annotation).
  - Yellow/spinner = currently computing.
  - Grey = queued (not yet started).
- When an SSE `ready` event arrives, update the dot without a full re-render.
- If the user clicks a queued or computing image, show `In queue… position N` instead
  of the elapsed-seconds counter; update to `Computing rings… Xs` once it starts.

### Tests

- `GET /` responds in < 2 s even when 25 images are stale (server starts before queue runs).
- After the queue drains, all filenames appear in `generated` with `algorithm_hash = currentHash`.
- `GET /api/events` with `Accept: text/event-stream` returns `Content-Type: text/event-stream`.
- SSE events are emitted for each image as it completes (test with a small mock image).

---

## AW-6 — Server-side log file for invalid / unexpected data

**Goal:** Persist a structured log of every anomaly the server encounters (corrupt DB rows,
detection failures, fallback triggers) to a file so problems can be diagnosed after the fact
without keeping a terminal open.

### Log file

Location: `logs/annotate.log` next to the repo root (created on first write, appended on
subsequent runs). Each entry is one JSON line (NDJSON):

```json
{"ts":"2026-03-24T21:00:00.000Z","level":"warn","event":"invalid_generated","filename":"foo.jpg","detail":"ring 3 has null coordinates"}
{"ts":"2026-03-24T21:00:05.000Z","level":"error","event":"detection_failed","filename":"bar.jpg","detail":"findTarget returned success=false: <error message>"}
{"ts":"2026-03-24T21:00:10.000Z","level":"info","event":"fallback_slow_path","filename":"foo.jpg","detail":"recomputing after invalid cache"}
```

### Events to log

| Event | Level | When |
|---|---|---|
| `invalid_generated` | warn | Fast path: stored data fails `isValidDetected()` |
| `fallback_slow_path` | info | Fast path fell back to slow path for any reason |
| `detection_failed` | error | `findTarget` returns `success: false` |
| `db_wipe` | warn | Startup migration wiped corrupt rows (with count) |
| `save_error` | error | `POST /api/save` throws |

### Server changes

```typescript
import * as fsSync from 'fs';

const LOG_PATH = path.resolve(__dirname, '../logs/annotate.log');
fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });

function logEvent(level: 'info'|'warn'|'error', event: string, filename: string, detail: string) {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, event, filename, detail });
  fsSync.appendFileSync(LOG_PATH, line + '\n');
  if (level === 'error') console.error(`[${level}] ${event}: ${filename} — ${detail}`);
  else if (level === 'warn') console.warn(`[${level}] ${event}: ${filename} — ${detail}`);
}
```

Call sites:
- AW-2 validity check: `logEvent('warn', 'invalid_generated', filename, 'ring N has null coordinates')` before deleting the row.
- `logEvent('info', 'fallback_slow_path', filename, reason)` whenever fast path falls through.
- Slow path: `logEvent('error', 'detection_failed', filename, result.error)` when `result.success === false`.
- Startup wipe: `logEvent('warn', 'db_wipe', '', `${rowCount} rows wiped`)`.

### Tests

- After starting the server and requesting an image that has a corrupt `generated` row, `logs/annotate.log` exists and contains a line with `"event":"invalid_generated"` for that filename.
- After a successful fast-path load, no error or warn lines are appended.
- Log file survives a server restart (append mode, not overwrite).

---

## Task status

| Task | Status | Dependencies |
|---|---|---|
| AW-1 Separate tables | done | — |
| AW-2 Fast path validity check | done | AW-1 |
| AW-3 Collapsible data panel | done | — |
| AW-4 Auto-deduce score from rings | done | — |
| AW-5 Non-blocking background generation | done | AW-1 |
| AW-6 Server-side log file | done | AW-2 |
