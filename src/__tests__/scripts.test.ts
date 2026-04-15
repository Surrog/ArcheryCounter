/**
 * Regression tests for npm scripts defined in package.json.
 *
 * Skipped scripts (not testable in CI):
 *   android, ios, start — require a connected device / simulator
 *   test               — this test runner itself
 *   lint               — pre-existing ft-flow / ESLint 9 compatibility failure
 *
 * DB-dependent tests (dump-annotations, seed-from-parquet, annotate) rely on
 * the same PostgreSQL instance used by groundTruth.test.ts.
 */

import { execSync, spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import * as crypto from 'crypto';
import { Pool } from 'pg';

const ROOT = path.resolve(__dirname, '../..');

/** Run an npm script synchronously; throws on non-zero exit. */
function npmRun(script: string, env?: NodeJS.ProcessEnv, timeoutMs = 60_000): string {
  return execSync(`npm run --silent ${script}`, {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: 'pipe',
    env: { ...process.env, ...env },
    timeout: timeoutMs,
  });
}

// ---------------------------------------------------------------------------
// postinstall
// ---------------------------------------------------------------------------

test('postinstall: patch-gradle-plugin exits 0 and reports status', () => {
  const stdout = execSync('node scripts/patch-gradle-plugin.js', {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: 'pipe',
  });
  // Each patched file prints "patched <file>" or "already patched <file>"
  expect(stdout).toMatch(/patch-gradle-plugin: (patched|already patched)/);
}, 15_000);

// ---------------------------------------------------------------------------
// dump-annotations
// ---------------------------------------------------------------------------

test('dump-annotations: exits 0 and writes parquet with arrows column', () => {
  const stdout = npmRun('dump-annotations', undefined, 30_000);
  expect(stdout.trim()).toMatch(/^Wrote \d+ rows →/);

  // Verify the parquet file exists and is non-empty
  const parquetPath = path.join(ROOT, 'data/annotations.parquet');
  expect(fs.existsSync(parquetPath)).toBe(true);
  expect(fs.statSync(parquetPath).size).toBeGreaterThan(1000);
}, 30_000);

// ---------------------------------------------------------------------------
// seed-from-parquet
// ---------------------------------------------------------------------------

test('seed-from-parquet: exits 0 and seeds rows from parquet', () => {
  const stdout = npmRun('seed-from-parquet', undefined, 30_000);
  expect(stdout.trim()).toMatch(/^Seeded \d+ rows from/);
}, 30_000);

// ---------------------------------------------------------------------------
// visualize
// ---------------------------------------------------------------------------

test('visualize: exits 0 and writes a valid report.html', () => {
  const reportPath = path.join(ROOT, 'report.html');
  if (fs.existsSync(reportPath)) fs.unlinkSync(reportPath);

  const stdout = npmRun('visualize -- --limit 3', { ...process.env, VISUALIZE_NO_NN: '1' }, 120_000);

  // Script prints "X/Y passed" summary
  expect(stdout).toMatch(/\d+\/\d+ passed/);

  expect(fs.existsSync(reportPath)).toBe(true);
  const html = fs.readFileSync(reportPath, 'utf8');
  expect(html).toContain('<!DOCTYPE html>');
  expect(html).toContain('ArcheryCounter');
}, 120_000);

// ---------------------------------------------------------------------------
// annotate (long-running HTTP server)
// ---------------------------------------------------------------------------

/** DB connection config — mirrors defaults in annotate.ts */
const DB_CONFIG = {
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
};

/** Compute the same algorithm hash the server uses. */
function computeAlgorithmHash(): string {
  const files = ['src/targetDetection.ts']
    .map(f => path.join(ROOT, f))
    .filter(f => fs.existsSync(f))
    .map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

/** Wait until the annotate server prints its ready message, then call cb. */
function waitForAnnotateReady(
  proc: ReturnType<typeof spawn>,
  port: number,
  timeoutMs: number,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('annotate server did not start within timeout')),
      timeoutMs,
    );
    proc.stdout?.on('data', (chunk: Buffer) => {
      if (chunk.toString().includes(`Annotation tool: http://localhost:${port}`)) {
        clearTimeout(timer);
        resolve();
      }
    });
    proc.on('error', (e) => { clearTimeout(timer); reject(e); });
    proc.on('exit', (code) => {
      if (code !== null && code !== 0) {
        clearTimeout(timer);
        reject(new Error(`annotate exited with code ${code}`));
      }
    });
  });
}

/** Find a free TCP port by asking the OS to assign one. */
function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = http.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as { port: number }).port;
      srv.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

test('annotate: server starts, listens, and serves GET /', async () => {
  const port = await getFreePort();
  return new Promise<void>((resolve, reject) => {
    const proc = spawn('npm', ['run', 'annotate'], {
      cwd: ROOT,
      env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let done = false;
    const finish = (err?: unknown) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try { proc.kill('SIGTERM'); } catch {}
      if (err) reject(err);
      else resolve();
    };

    const timer = setTimeout(
      () => finish(new Error('annotate server did not start within timeout')),
      300_000,
    );

    proc.stdout?.on('data', (chunk: Buffer) => {
      if (chunk.toString().includes(`Annotation tool: http://localhost:${port}`)) {
        const req = http.get(`http://localhost:${port}/`, (res) => {
          let body = '';
          res.on('data', (c: string) => { body += c; });
          res.on('end', () => {
            try {
              expect(res.statusCode).toBe(200);
              expect(body).toContain('<!DOCTYPE html>');
              finish();
            } catch (e) { finish(e); }
          });
        });
        req.on('error', (e) => finish(e));
      }
    });

    proc.on('error', (e) => finish(e));
    proc.on('exit', (code) => {
      // null code means killed by signal (expected from our SIGTERM)
      if (code !== null && code !== 0) finish(new Error(`annotate exited with code ${code}`));
    });
  });
}, 300_000);

// ---------------------------------------------------------------------------
// annotate: corrupt generated data triggers fallback + recompute
// ---------------------------------------------------------------------------

test('annotate: corrupt generated row is deleted and image is recomputed correctly', async () => {
  const FILENAME = '20190321_211008.jpg'; // always present; ~45 s to process
  const currentHash = computeAlgorithmHash();
  const CORRUPT_RINGS = JSON.stringify([{ points: [[null, null], [null, null], [null, null]] }]);

  // 1. Inject a corrupt generated row with the correct hash so the server
  //    will think the image is ready and attempt the fast path.
  const db = new Pool(DB_CONFIG);
  try {
    await db.query(`
      INSERT INTO generated (filename, algorithm_hash, rings, arrows)
      VALUES ($1, $2, $3, '[]')
      ON CONFLICT (filename) DO UPDATE
        SET algorithm_hash = EXCLUDED.algorithm_hash,
            rings          = EXCLUDED.rings
    `, [FILENAME, currentHash, CORRUPT_RINGS]);
  } finally {
    await db.end();
  }

  // 2. Start the server — it loads inGenerated and sees the corrupt row as "ready".
  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // 3. Request the image. Fast path reads corrupt rings → AW-2 validity check fails
    //    → deletes row → falls back to slow path → recomputes.
    const imageRes = await fetch(`http://localhost:${port}/api/image/${encodeURIComponent(FILENAME)}`);
    expect(imageRes.status).toBe(200);
    const data = await imageRes.json() as {
      detected: { targets: { paperBoundary: [number,number][]; ringSets: { points: [number,number][] }[][] }[]; arrows: unknown[] };
    };

    // 4. Verify the response contains valid ring data (no null coordinates).
    const rings = data.detected.targets?.[0]?.ringSets?.[0] ?? [];
    expect(rings.length).toBeGreaterThan(0);
    for (const ring of rings) {
      for (const pt of ring.points) {
        expect(pt[0]).not.toBeNull();
        expect(pt[1]).not.toBeNull();
        expect(typeof pt[0]).toBe('number');
      }
    }
    expect(data.detected.targets?.[0]?.paperBoundary?.length).toBeGreaterThan(0);

    // 5. Verify the DB generated row has been replaced with valid data.
    const db2 = new Pool(DB_CONFIG);
    try {
      const { rows } = await db2.query(
        'SELECT rings, algorithm_hash FROM generated WHERE filename = $1',
        [FILENAME],
      );
      expect(rows).toHaveLength(1);
      expect(rows[0].algorithm_hash).toBe(currentHash);
      expect(rows[0].rings.length).toBeGreaterThan(0);
      expect(rows[0].rings[0][0][0].points[0][0]).not.toBeNull();
    } finally {
      await db2.end();
    }

    // 6. Verify the log file recorded the invalid_generated and fallback events.
    const logPath = path.join(ROOT, 'logs/annotate.log');
    expect(fs.existsSync(logPath)).toBe(true);
    const logLines = fs.readFileSync(logPath, 'utf8')
      .trim().split('\n').filter(Boolean)
      .map(l => JSON.parse(l) as { event: string; filename: string });
    expect(logLines.some(l => l.event === 'invalid_generated' && l.filename === FILENAME)).toBe(true);
    expect(logLines.some(l => l.event === 'fallback_slow_path'  && l.filename === FILENAME)).toBe(true);

    // 7. Verify generation-status reflects the image as ready.
    const statusRes = await fetch(`http://localhost:${port}/api/generation-status`);
    const status = await statusRes.json() as Record<string, string>;
    expect(status[FILENAME]).toBe('ready');

  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 300_000);

// ---------------------------------------------------------------------------
// annotate: old flat SplineRing[] format triggers fallback + recompute
// ---------------------------------------------------------------------------

test('annotate: old flat rings format (pre-multi-target) triggers fallback and recompute', async () => {
  const FILENAME = '20190321_211008.jpg';
  const currentHash = computeAlgorithmHash();

  // Build an old-format row: flat SplineRing[] with valid points, stored directly
  // as the rings column value — this is what rows written before the multi-target
  // refactor look like.  dbToTargets reads this as 10 TargetData entries whose
  // ringSets field is a SplineRing object (not an array).  isValidDetected sees
  // Array.isArray(t.ringSets) === false and returns false → fallback to slow path.
  const OLD_FORMAT_RINGS = JSON.stringify(
    Array.from({ length: 10 }, (_, i) => ({
      points: Array.from({ length: 8 }, (__, j) => [
        Math.round(100 + i * 20 + j * 5),
        Math.round(100 + i * 20 + j * 5),
      ]),
    })),
  );

  const db = new Pool(DB_CONFIG);
  try {
    await db.query(`
      INSERT INTO generated (filename, algorithm_hash, rings, arrows)
      VALUES ($1, $2, $3, '[]')
      ON CONFLICT (filename) DO UPDATE
        SET algorithm_hash = EXCLUDED.algorithm_hash,
            rings          = EXCLUDED.rings
    `, [FILENAME, currentHash, OLD_FORMAT_RINGS]);
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // Fast path reads old-format row → isValidDetected fails → fallback recomputes.
    const imageRes = await fetch(`http://localhost:${port}/api/image/${encodeURIComponent(FILENAME)}`);
    expect(imageRes.status).toBe(200);
    const data = await imageRes.json() as {
      detected: { targets: { paperBoundary: [number,number][]; ringSets: { points: [number,number][] }[][] }[]; arrows: unknown[] };
    };

    // After recompute, rings must be valid.
    const rings = data.detected.targets?.[0]?.ringSets?.[0] ?? [];
    expect(rings.length).toBeGreaterThan(0);
    for (const ring of rings) {
      for (const pt of ring.points) {
        expect(pt[0]).not.toBeNull();
        expect(typeof pt[0]).toBe('number');
      }
    }

    // Log must record invalid_generated for this filename.
    const logPath = path.join(ROOT, 'logs/annotate.log');
    expect(fs.existsSync(logPath)).toBe(true);
    const logLines = fs.readFileSync(logPath, 'utf8')
      .trim().split('\n').filter(Boolean)
      .map(l => JSON.parse(l) as { event: string; filename: string });
    expect(logLines.some(l => l.event === 'invalid_generated' && l.filename === FILENAME)).toBe(true);
  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 300_000);

// ---------------------------------------------------------------------------
// annotate: all-zero paper_boundary in annotations is healed at read time
// ---------------------------------------------------------------------------

test('annotate: zero boundary in annotations is deleted (marked not-annotated) at read time', async () => {
  const FILENAME = '20190321_211008.jpg';
  const currentHash = computeAlgorithmHash();

  // 1. Put a valid boundary in generated and an all-zero boundary in annotations.
  const VALID_BOUNDARY = JSON.stringify([[[100, 200], [500, 100], [600, 800], [50, 750]]]);
  const ZERO_BOUNDARY  = JSON.stringify([[[0, 0], [0, 0], [0, 0], [0, 0]]]);

  const db = new Pool(DB_CONFIG);
  try {
    await db.query(`
      INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows)
      VALUES ($1, $2, $3, '[[[]]]', '[]')
      ON CONFLICT (filename) DO UPDATE
        SET algorithm_hash = EXCLUDED.algorithm_hash,
            paper_boundary = EXCLUDED.paper_boundary
    `, [FILENAME, currentHash, VALID_BOUNDARY]);

    await db.query(`
      INSERT INTO annotations (filename, paper_boundary, rings, arrows)
      VALUES ($1, $2, '[[[]]]', '[]')
      ON CONFLICT (filename) DO UPDATE
        SET paper_boundary = EXCLUDED.paper_boundary
    `, [FILENAME, ZERO_BOUNDARY]);
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // /api/annotations bulk endpoint must NOT include this image — the all-zero
    // boundary annotation is invalid and should be deleted, making it appear
    // as not-annotated so the user can re-annotate it.
    const bulkRes = await fetch(`http://localhost:${port}/api/annotations`);
    expect(bulkRes.status).toBe(200);
    const bulk = await bulkRes.json() as Record<string, unknown>;
    expect(bulk[FILENAME]).toBeUndefined();

    // /api/annotation/{filename} per-image endpoint returns 404 (no annotation).
    const annRes = await fetch(`http://localhost:${port}/api/annotation/${encodeURIComponent(FILENAME)}`);
    expect(annRes.status).toBe(404);
  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 60_000);

// ---------------------------------------------------------------------------
// annotate: all-zero boundary in generated triggers recompute (isValidDetected)
// ---------------------------------------------------------------------------

test('annotate: all-zero paper_boundary in generated triggers recompute', async () => {
  const FILENAME = '20190321_211008.jpg';
  const currentHash = computeAlgorithmHash();

  // Inject a generated row with current hash but all-zero boundary.
  const ZERO_BOUNDARY = JSON.stringify([[[0, 0], [0, 0], [0, 0], [0, 0]]]);
  const ZERO_RINGS    = JSON.stringify([[[{ points: Array.from({ length: 8 }, () => [0, 0]) }]]]);

  const db = new Pool(DB_CONFIG);
  try {
    await db.query(`
      INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows)
      VALUES ($1, $2, $3, $4, '[]')
      ON CONFLICT (filename) DO UPDATE
        SET algorithm_hash = EXCLUDED.algorithm_hash,
            paper_boundary = EXCLUDED.paper_boundary,
            rings          = EXCLUDED.rings
    `, [FILENAME, currentHash, ZERO_BOUNDARY, ZERO_RINGS]);
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // Fast path loads the zero boundary → isValidDetected returns false
    // → server falls back to slow path and recomputes.
    const imageRes = await fetch(`http://localhost:${port}/api/image/${encodeURIComponent(FILENAME)}`);
    expect(imageRes.status).toBe(200);
    const data = await imageRes.json() as {
      detected: { targets: { paperBoundary: [number,number][]; ringSets: { points: [number,number][] }[][] }[]; arrows: unknown[] };
    };

    // After recompute, rings must have valid (non-zero) points.
    const rings = data.detected.targets?.[0]?.ringSets?.[0] ?? [];
    expect(rings.length).toBeGreaterThan(0);
    for (const ring of rings) {
      expect(ring.points.some(([x, y]) => x !== 0 || y !== 0)).toBe(true);
    }

    // Log must record invalid_generated for this filename.
    const logPath = path.join(ROOT, 'logs/annotate.log');
    const logLines = fs.existsSync(logPath)
      ? fs.readFileSync(logPath, 'utf8').trim().split('\n').filter(Boolean)
          .map(l => JSON.parse(l) as { event: string; filename: string })
      : [];
    expect(logLines.some(l => l.event === 'invalid_generated' && l.filename === FILENAME)).toBe(true);
  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 300_000);

// ---------------------------------------------------------------------------
// annotate: wrapSingleTarget rejects all-zero boundary (unit-level via server)
// ---------------------------------------------------------------------------

test('annotate: detection with all-zero boundary stores empty boundary in generated', async () => {
  // Verify that after a forced recompute the server never writes an all-zero
  // boundary to generated: inject a stale-hash row so recompute is triggered,
  // wait for background processing, then verify generated.paper_boundary is
  // either empty ([]) or has at least one non-zero vertex.
  const FILENAME = '20190321_211008.jpg';
  const currentHash = computeAlgorithmHash();

  const db = new Pool(DB_CONFIG);
  try {
    await db.query('DELETE FROM generated WHERE filename = $1', [FILENAME]);
    await db.query('DELETE FROM annotations WHERE filename = $1', [FILENAME]);
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // Request the image to trigger slow-path recompute.
    const imageRes = await fetch(`http://localhost:${port}/api/image/${encodeURIComponent(FILENAME)}`);
    expect(imageRes.status).toBe(200);

    // Check what was stored in generated.
    const db2 = new Pool(DB_CONFIG);
    try {
      const { rows } = await db2.query(
        'SELECT algorithm_hash, paper_boundary FROM generated WHERE filename = $1',
        [FILENAME],
      );
      expect(rows).toHaveLength(1);
      expect(rows[0].algorithm_hash).toBe(currentHash);

      // paper_boundary must be either absent ([]) or non-zero.
      const pb = rows[0].paper_boundary as [number,number][][] | null;
      if (pb && pb.length > 0 && pb[0].length > 0) {
        expect(pb[0].some(([x, y]) => x !== 0 || y !== 0)).toBe(true);
      }
    } finally {
      await db2.end();
    }
  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 300_000);

// ---------------------------------------------------------------------------
// annotate: generation status transitions (queued → computing → ready)
// ---------------------------------------------------------------------------

test('annotate: generation status transitions from queued to ready, SSE event received', async () => {
  const FILENAME = '20190321_211008.jpg'; // first alphabetically → processed first by background queue
  const currentHash = computeAlgorithmHash();

  // 1. Set up a clean state: only our test image in the generated table with a stale hash
  //    so the background queue has exactly one image to process (deterministic timing).
  const db = new Pool(DB_CONFIG);
  try {
    await db.query('DELETE FROM generated');
    await db.query(
      `INSERT INTO generated (filename, algorithm_hash, rings, arrows) VALUES ($1, 'stale_for_test', '[]', '[]')`,
      [FILENAME],
    );
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // 2. Subscribe to SSE before checking anything so we don't miss an early event.
    //    Collect all status events for FILENAME.
    const receivedStates: string[] = [];
    let sseResolve: (() => void) | null = null;
    const sseReady = new Promise<void>((resolve) => { sseResolve = resolve; });

    const sseReq = http.get(`http://localhost:${port}/api/events`, (res) => {
      let buf = '';
      res.on('data', (chunk: Buffer) => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const msg = JSON.parse(line.slice(6)) as { type?: string; filename?: string; state?: string };
            if (msg.type === 'status' && msg.filename === FILENAME && msg.state) {
              receivedStates.push(msg.state);
              if (msg.state === 'ready') sseResolve?.();
            }
          } catch {}
        }
      });
    });
    sseReq.on('error', () => {}); // silently ignore cleanup errors

    // 3. Initial status must be queued (or computing if the queue started very fast).
    const status0 = await fetch(`http://localhost:${port}/api/generation-status`).then(r => r.json()) as Record<string, string>;
    expect(['queued', 'computing']).toContain(status0[FILENAME]);

    // /api/image-status should also report stale (not yet processed with current hash).
    const imgStatus0 = await fetch(`http://localhost:${port}/api/image-status/${encodeURIComponent(FILENAME)}`).then(r => r.json()) as { state: string };
    expect(imgStatus0.state).toBe('stale');

    // 4. Wait for the SSE "ready" event (background queue finishes processing the image).
    await Promise.race([
      sseReady,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('SSE ready event not received within 120 s')), 120_000),
      ),
    ]);
    sseReq.destroy();

    // 5. The SSE stream must have included a "ready" state for our file.
    expect(receivedStates).toContain('ready');

    // 6. /api/generation-status now reports ready.
    const status1 = await fetch(`http://localhost:${port}/api/generation-status`).then(r => r.json()) as Record<string, string>;
    expect(status1[FILENAME]).toBe('ready');

    // 7. /api/image-status also reports ready.
    const imgStatus1 = await fetch(`http://localhost:${port}/api/image-status/${encodeURIComponent(FILENAME)}`).then(r => r.json()) as { state: string };
    expect(imgStatus1.state).toBe('ready');

    // 8. The generated row in the DB now has the correct hash and non-empty rings.
    const db2 = new Pool(DB_CONFIG);
    try {
      const { rows } = await db2.query(
        'SELECT algorithm_hash, rings FROM generated WHERE filename = $1',
        [FILENAME],
      );
      expect(rows).toHaveLength(1);
      expect(rows[0].algorithm_hash).toBe(currentHash);
      expect(rows[0].rings.length).toBeGreaterThan(0);
    } finally {
      await db2.end();
    }

  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 180_000);

// ---------------------------------------------------------------------------
// annotate: startup wipes corrupt annotation rows, preserves valid ones
// ---------------------------------------------------------------------------

test('annotate: startup wipes corrupt annotation rows; invalid annotations deleted on load; valid ones preserved', async () => {
  const BAD_FILE  = '20260319_213758.jpg'; // corrupt rings (null coords) → wiped → deleted as invalid
  const GOOD_FILE = '20190321_212956.jpg'; // full valid annotation → preserved

  // Corrupt rings matching the old buggy code output (null coordinates)
  const CORRUPT_RINGS = JSON.stringify([
    { points: [[null, null], [null, null], [null, null]] },
  ]);
  // Valid annotation: 10 rings (required by isValidAnnotation), each with >= 3 points
  const VALID_RINGS = JSON.stringify(
    Array.from({ length: 10 }, (_, i) => {
      const r = 20 + i * 22;
      return { points: [[500 + r, 400], [500, 400 + r], [500 - r, 400], [500, 400 - r]] };
    }),
  );
  const VALID_BOUNDARY = JSON.stringify([[50, 50], [950, 50], [950, 750], [50, 750]]);
  const VALID_ARROWS   = JSON.stringify([{ tip: [500, 400], nock: null, score: 9 }]);

  // 1. Inject known state so the test is deterministic.
  const db = new Pool(DB_CONFIG);
  try {
    await db.query(
      `INSERT INTO annotations (filename, rings, paper_boundary, arrows)
       VALUES ($1, $2, NULL, '[]')
       ON CONFLICT (filename) DO UPDATE SET rings = EXCLUDED.rings, paper_boundary = NULL, arrows = '[]'`,
      [BAD_FILE, CORRUPT_RINGS],
    );
    await db.query(
      `INSERT INTO annotations (filename, rings, paper_boundary, arrows)
       VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb)
       ON CONFLICT (filename) DO UPDATE
         SET rings = EXCLUDED.rings, paper_boundary = EXCLUDED.paper_boundary, arrows = EXCLUDED.arrows`,
      [GOOD_FILE, VALID_RINGS, VALID_BOUNDARY, VALID_ARROWS],
    );
  } finally {
    await db.end();
  }

  // 2. Start the server — startup wipes corrupt rows, then GET /api/annotations deletes invalid ones.
  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // 3. Fetch all annotations from the server.
    const res = await fetch(`http://localhost:${port}/api/annotations`);
    expect(res.status).toBe(200);
    const annotations = await res.json() as Record<string, {
      targets: { paperBoundary: [number, number][]; ringSets: unknown[][] }[];
      arrows: { tip: [number, number] }[];
    }>;

    // 4. Bad file: startup wipe resets its rings to [] → becomes invalid → deleted on load.
    expect(annotations[BAD_FILE]).toBeUndefined();

    // 5. Good file: valid annotation must be untouched (has arrows → passes isValidAnnotation).
    expect(annotations[GOOD_FILE]).toBeDefined();
    expect(annotations[GOOD_FILE].arrows.length).toBeGreaterThan(0);
    expect(annotations[GOOD_FILE].targets.length).toBeGreaterThan(0);

    // 6. The log must record a db_wipe event (at least one corrupt row was wiped).
    const logPath = path.join(ROOT, 'logs/annotate.log');
    expect(fs.existsSync(logPath)).toBe(true);
    const logLines = fs.readFileSync(logPath, 'utf8')
      .trim().split('\n').filter(Boolean)
      .map(l => JSON.parse(l) as { event: string });
    expect(logLines.some(l => l.event === 'db_wipe')).toBe(true);

  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 60_000);

// ---------------------------------------------------------------------------
// annotate: invalid annotation validation
// ---------------------------------------------------------------------------

test('annotate: GET /api/annotations deletes incomplete annotations; POST /api/save rejects invalid ones', async () => {
  const INCOMPLETE_FILE = '20260319_213758.jpg'; // will be seeded with only 9 rings (< 10) → invalid
  const VALID_FILE      = '20190321_212956.jpg'; // will be seeded with full valid annotation

  // isValidAnnotation requires exactly 10 rings, each with >= 3 points, plus a boundary.
  // INCOMPLETE_RINGS has only 9 rings → invalid.
  const INCOMPLETE_RINGS = JSON.stringify(
    Array.from({ length: 9 }, (_, i) => {
      const r = 20 + i * 22;
      return { points: [[500 + r, 400], [500, 400 + r], [500 - r, 400], [500, 400 - r]] };
    }),
  );
  const RINGS = JSON.stringify(
    Array.from({ length: 10 }, (_, i) => {
      const r = 20 + i * 22;
      return { points: [[500 + r, 400], [500, 400 + r], [500 - r, 400], [500, 400 - r]] };
    }),
  );
  const BOUNDARY = JSON.stringify([[50, 50], [950, 50], [950, 750], [50, 750]]);
  const ARROWS   = JSON.stringify([{ tip: [500, 400], nock: null, score: 9 }]);

  // Seed: INCOMPLETE_FILE has only 9 rings → invalid (isValidAnnotation requires 10)
  //        VALID_FILE has all three components → valid
  const db = new Pool(DB_CONFIG);
  try {
    await db.query(
      `INSERT INTO annotations (filename, rings, paper_boundary, arrows)
       VALUES ($1, $2::jsonb, $3::jsonb, '[]')
       ON CONFLICT (filename) DO UPDATE
         SET rings = EXCLUDED.rings, paper_boundary = EXCLUDED.paper_boundary, arrows = '[]'`,
      [INCOMPLETE_FILE, INCOMPLETE_RINGS, BOUNDARY],
    );
    await db.query(
      `INSERT INTO annotations (filename, rings, paper_boundary, arrows)
       VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb)
       ON CONFLICT (filename) DO UPDATE
         SET rings = EXCLUDED.rings, paper_boundary = EXCLUDED.paper_boundary, arrows = EXCLUDED.arrows`,
      [VALID_FILE, RINGS, BOUNDARY, ARROWS],
    );
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // 1. GET /api/annotations: incomplete annotation is deleted, valid one is returned.
    const getRes = await fetch(`http://localhost:${port}/api/annotations`);
    expect(getRes.status).toBe(200);
    const annotations = await getRes.json() as Record<string, unknown>;
    expect(annotations[INCOMPLETE_FILE]).toBeUndefined();
    expect(annotations[VALID_FILE]).toBeDefined();

    // 2. POST /api/save with an invalid annotation should not persist it.
    const saveRes = await fetch(`http://localhost:${port}/api/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        [INCOMPLETE_FILE]: {
          paperBoundary: [[50, 50], [950, 50], [950, 750], [50, 750]],
          rings: JSON.parse(INCOMPLETE_RINGS), // invalid: only 9 rings
          arrows: [{ tip: [500, 400], nock: null, score: 9 }],
        },
      }),
    });
    expect(saveRes.status).toBe(200);

    // Verify the invalid annotation was not saved.
    const getRes2 = await fetch(`http://localhost:${port}/api/annotations`);
    const annotations2 = await getRes2.json() as Record<string, unknown>;
    expect(annotations2[INCOMPLETE_FILE]).toBeUndefined();

  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 60_000);

// ---------------------------------------------------------------------------
// annotate: /api/save returns well-formed JSON even on parse error
// ---------------------------------------------------------------------------

test('annotate: /api/save returns valid JSON with error key on malformed body', async () => {
  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // Send a body that is not valid JSON (previously produced {"error":"SyntaxError: ..."} without
    // proper escaping — this test guards against regression to the `{"error":"${e}"}` template).
    const res = await fetch(`http://localhost:${port}/api/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{invalid json: "unterminated',
    });

    expect(res.status).toBe(500);

    // Must parse as valid JSON.
    const json = await res.json() as { error: unknown };
    expect(typeof json.error).toBe('string');
    expect((json.error as string).length).toBeGreaterThan(0);
  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 60_000);

// ---------------------------------------------------------------------------
// annotate: concurrent /api/image/ requests for same file return same data
// ---------------------------------------------------------------------------

test('annotate: concurrent /api/image/ requests for same cached file both return 200', async () => {
  const FILENAME = '20190321_211008.jpg';
  const currentHash = computeAlgorithmHash();

  // Pre-seed a ready generated row so the fast path is taken (no 45s recompute).
  const db = new Pool(DB_CONFIG);
  try {
    // Ensure a valid generated row exists with the current hash.
    await db.query(`
      INSERT INTO generated (filename, algorithm_hash, rings, arrows)
      VALUES ($1, $2, '[[]]', '[]')
      ON CONFLICT (filename) DO UPDATE
        SET algorithm_hash = EXCLUDED.algorithm_hash
    `, [FILENAME, currentHash]);
  } finally {
    await db.end();
  }

  const port = await getFreePort();
  const proc = spawn('npm', ['run', 'annotate'], {
    cwd: ROOT,
    env: { ...process.env, NO_BROWSER: '1', ANNOTATE_PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForAnnotateReady(proc, port, 30_000);

    // Fire two concurrent requests for the same uncached image.
    const url = `http://localhost:${port}/api/image/${encodeURIComponent(FILENAME)}`;
    const [res1, res2] = await Promise.all([fetch(url), fetch(url)]);

    expect(res1.status).toBe(200);
    expect(res2.status).toBe(200);

    // Both responses must be valid JSON.
    const [body1, body2] = await Promise.all([res1.json(), res2.json()]);
    expect(typeof body1).toBe('object');
    expect(typeof body2).toBe('object');
  } finally {
    try { proc.kill('SIGTERM'); } catch {}
  }
}, 60_000);
