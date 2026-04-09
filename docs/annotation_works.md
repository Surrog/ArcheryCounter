# Annotation Tool — Features

All tasks completed. This document is a reference summary of what was built in
`scripts/annotate.ts`.

---

## Architecture

- Node.js HTTP server (`scripts/annotate.ts`) serves a browser UI and a REST API.
- PostgreSQL backend: `annotations` table (human data) + `generated` table (algorithm output).
- Detection runs in a background queue on startup; browser receives SSE events as images complete.

---

## DB schema

```sql
-- human annotations
annotations(filename TEXT PK, paper_boundary JSONB, rings JSONB, arrows JSONB DEFAULT '[]')

-- algorithm output (never overwritten by save)
generated(filename TEXT PK, algorithm_hash TEXT, paper_boundary JSONB, rings JSONB, arrows JSONB, updated_at TIMESTAMPTZ)
```

---

## Keyboard / interaction reference

| Action | Shortcut |
|---|---|
| Add boundary vertex | Ctrl + click on canvas |
| Add ring control point | Alt + click near ring curve |
| Remove vertex / ring point / arrow endpoint | Shift + click handle |
| Add arrow (tip) | A key, or "Add arrow" button |
| Score picker (after placing tip) | Keys 1–9, X (inner gold), M (miss), Esc (null) |
| Toggle ring overlay + handles | "Rings" checkbox |
| Toggle boundary overlay + handles | "Boundary" checkbox |
| Toggle arrows layer | "Arrows" checkbox |
| Save and go to next image | N key |
| Save | Save button (shows reason when disabled) |

---

## Completed features (AW-1 – AW-9)

| Feature | Notes |
|---|---|
| Separate annotations / generated tables | Save never overwrites detection cache |
| Fast-path validity check | Corrupt cached rows are deleted and recomputed |
| Collapsible data panel | State persisted in localStorage |
| Auto-deduce arrow score from rings | Suggested score highlighted in picker |
| Non-blocking background generation | SSE status per image; server starts immediately |
| Server-side log file | `logs/annotate.log` (NDJSON, append) |
| Ring control point insertion | Alt+click inserts point on nearest spline segment |
| Save button feedback | Inline message explains what's missing when save is disabled |
| Client + server save logging | Console logs on both sides for diagnosing save failures |
| Delete image (AW-7) | `DELETE /api/image/:filename`; removes file + DB rows + caches; SSE broadcast |
| Remove nock annotation (AW-8) | State machine simplified to `idle → place-tip → score-input`; DB migration script (`migrate-remove-nock.ts`) removes stale nock fields |
| Floating score picker (AW-9) | `position: fixed` overlay appearing near the tip; centered on click, clamped to ±50 px vertically |
| Handles merged into layer toggles | Rings checkbox controls ring splines + handles; Boundary checkbox controls boundary polygon + handles; standalone Handles checkbox removed |
| N shortcut saves and advances | Pressing N saves current annotations then moves to the next image |
| Boundary coordinate clamping | All `paperBoundary` points clamped to `[0, W-1] × [0, H-1]` at source (`targetDetection.ts`), on every server insert path, and in the save endpoint; `migrate-clamp-boundary.ts` fixed existing DB rows |
