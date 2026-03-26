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
| Add arrow (tip then nock) | A key, or "Add arrow" button |
| Score picker (after placing nock) | Keys 1–9, X (inner gold), M (miss), Esc (null) |
| Toggle arrows layer | "Arrows" checkbox |
| Save | Save button (shows reason when disabled) |

---

## Completed features (AW-1 – AW-6)

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
