---
name: Test architecture — generated table + globalSetup
description: How tests are structured to run fast via pre-computed DB cache
type: project
---

the detection algorithm inline.

**Key files:**
- `src/__tests__/globalSetup.ts`: Runs once before all test suites (Jest `globalSetup`), detects all images in `images/` using a process pool (CONCURRENCY = min(cpus, 4)), stores results in `generated` table. Skips images whose `algorithm_hash` matches current. Forces re-detection when algorithm files change.
- `generated` table columns: `filename, algorithm_hash, paper_boundary JSONB, rings JSONB, arrows JSONB, width INT, height INT, updated_at`

**Test timing (2026-03-28):**
- globalSetup: ~225s first run (all stale), ~0s subsequent runs
- groundTruth.test.ts: ~3s (DB queries only)
- targetDetection.test.ts: ~5s (DB queries) + ~45s (1 structural inline test)
- scripts.test.ts: ~90s (starts annotate server)
- Total: ~93s ✓

**Why:** Previously targetDetection ran detection inline (~45s/image × 20 images = ~400s). Restructured to read from generated table. Scripts test still slow because it spawns the annotate server.

**Algorithm hash:** SHA-256 of `src/targetDetection.ts` + `src/arrowDetection.ts`, hex[:16]. When either file changes, all images are re-detected on next test run.

**How to apply:** When adding new test quality checks, use the generated table instead of inline detection. Keep the structural test (`ringPoints`/`rayDebug`) inline on 1 representative image.
