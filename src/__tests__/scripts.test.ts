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

  const stdout = npmRun('visualize', undefined, 300_000);

  // Script prints "X/Y passed" summary
  expect(stdout).toMatch(/\d+\/\d+ passed/);

  expect(fs.existsSync(reportPath)).toBe(true);
  const html = fs.readFileSync(reportPath, 'utf8');
  expect(html).toContain('<!DOCTYPE html>');
  expect(html).toContain('ArcheryCounter');
}, 300_000);

// ---------------------------------------------------------------------------
// annotate (long-running HTTP server)
// ---------------------------------------------------------------------------

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
