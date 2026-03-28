/**
 * Jest globalSetup: runs once before all test suites.
 *
 * Populates the `generated` table with detection results for all images in
 * images/.  Individual test suites then read from this table (fast DB query)
 * rather than running the detection algorithm inline.
 */
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
import { Pool } from 'pg';

const ROOT         = path.resolve(__dirname, '../..');
const IMAGES_DIR   = path.join(ROOT, 'images');
const TSX_BIN      = path.join(ROOT, 'node_modules/.bin/tsx');
const DETECT_WORKER = path.join(ROOT, 'scripts/detect-worker.ts');
const CONCURRENCY  = Math.min(os.cpus().length, 4);

function computeAlgorithmHash(): string {
  const files = [
    path.join(ROOT, 'src/targetDetection.ts'),
    path.join(ROOT, 'src/arrowDetection.ts'),
  ].filter(f => fs.existsSync(f)).map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

function runWorker(imgPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn(TSX_BIN, [DETECT_WORKER, imgPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '', stderr = '';
    proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });
    const timer = setTimeout(() => {
      try { proc.kill('SIGTERM'); } catch {}
      reject(new Error('Worker timed out'));
    }, 5 * 60 * 1000);
    proc.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(`Worker exited ${code}: ${stderr.slice(0, 300)}`)); return; }
      try { resolve(JSON.parse(stdout)); }
      catch { reject(new Error(`Bad JSON: ${stdout.slice(0, 200)}`)); }
    });
  });
}

export default async function globalSetup(): Promise<void> {
  const db = new Pool({
    host:     process.env.DB_HOST     || 'localhost',
    port:     parseInt(process.env.DB_PORT || '5432'),
    user:     process.env.DB_USER     || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    database: process.env.DB_NAME     || 'postgres',
  });

  try {
    await db.query(`
      CREATE TABLE IF NOT EXISTS generated (
        filename       TEXT PRIMARY KEY,
        algorithm_hash TEXT NOT NULL,
        paper_boundary JSONB,
        rings          JSONB NOT NULL DEFAULT '[]',
        arrows         JSONB NOT NULL DEFAULT '[]',
        width          INT,
        height         INT,
        updated_at     TIMESTAMPTZ DEFAULT NOW()
      )
    `);
    await db.query(`ALTER TABLE generated ADD COLUMN IF NOT EXISTS width INT`);
    await db.query(`ALTER TABLE generated ADD COLUMN IF NOT EXISTS height INT`);

    const currentHash = computeAlgorithmHash();
    const { rows } = await db.query('SELECT filename, algorithm_hash FROM generated');
    const inGenerated = new Map<string, string>(
      rows.map((r: any) => [r.filename as string, r.algorithm_hash as string]),
    );

    const imageFiles = fs
      .readdirSync(IMAGES_DIR)
      .filter(f => /\.(jpg|jpeg)$/i.test(f))
      .sort();

    const stale = imageFiles.filter(f => inGenerated.get(f) !== currentHash);
    if (stale.length === 0) {
      console.log('[globalSetup] All images up to date.');
      return;
    }

    console.log(`[globalSetup] Detecting ${stale.length} image(s) with ${CONCURRENCY} workers…`);
    let done = 0;

    for (let i = 0; i < stale.length; i += CONCURRENCY) {
      const batch = stale.slice(i, i + CONCURRENCY);
      await Promise.all(batch.map(async filename => {
        const imgPath = path.join(IMAGES_DIR, filename);
        try {
          const result = await runWorker(imgPath);
          await db.query(
            `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             ON CONFLICT (filename) DO UPDATE
               SET algorithm_hash = EXCLUDED.algorithm_hash,
                   paper_boundary = EXCLUDED.paper_boundary,
                   rings          = EXCLUDED.rings,
                   arrows         = EXCLUDED.arrows,
                   width          = EXCLUDED.width,
                   height         = EXCLUDED.height,
                   updated_at     = NOW()`,
            [filename, currentHash,
             JSON.stringify(result.paperBoundary),
             JSON.stringify(result.rings),
             JSON.stringify(result.arrows),
             result.width ?? null,
             result.height ?? null],
          );
          done++;
          console.log(`[globalSetup]   [${done}/${stale.length}] ${filename}`);
        } catch (err) {
          console.error(`[globalSetup]   FAILED ${filename}: ${err}`);
        }
      }));
    }
  } finally {
    await db.end();
  }
}
