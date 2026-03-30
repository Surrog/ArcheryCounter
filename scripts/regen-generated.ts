#!/usr/bin/env tsx
/**
 * Batch-regenerates the `generated` table for all annotated images whose
 * algorithm_hash is stale (or missing). Runs detect-worker in parallel.
 */
import * as path from 'path';
import * as fs from 'fs';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
import { Pool } from 'pg';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const DETECT_WORKER = path.resolve(__dirname, 'detect-worker.ts');
const CONCURRENCY = 4;

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

function computeAlgorithmHash(): string {
  const files = [
    path.resolve(__dirname, '../src/targetDetection.ts'),
    path.resolve(__dirname, '../src/arrowDetection.ts'),
  ].filter(f => fs.existsSync(f)).map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

function runDetectWorker(imgPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const child = spawn('npx', ['tsx', DETECT_WORKER, imgPath], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    child.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    child.stderr.on('data', (d: Buffer) => { /* suppress */ });
    child.on('close', (code) => {
      if (code !== 0) return reject(new Error(`detect-worker exited ${code} for ${imgPath}`));
      try { resolve(JSON.parse(stdout)); }
      catch (e) { reject(new Error(`JSON parse error for ${imgPath}: ${e}`)); }
    });
    child.on('error', reject);
  });
}

async function processImage(filename: string, currentHash: string): Promise<void> {
  const imgPath = path.join(IMAGES_DIR, filename);
  try {
    const result = await runDetectWorker(imgPath);
    await db.query(
      `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (filename) DO UPDATE
         SET algorithm_hash = EXCLUDED.algorithm_hash,
             paper_boundary = EXCLUDED.paper_boundary,
             rings          = EXCLUDED.rings,
             arrows         = EXCLUDED.arrows,
             width          = EXCLUDED.width,
             height         = EXCLUDED.height`,
      [filename, currentHash,
       result.paperBoundary ? JSON.stringify(result.paperBoundary) : null,
       JSON.stringify(result.rings),
       JSON.stringify(result.arrows),
       result.width, result.height],
    );
    console.log(`  OK  ${filename}`);
  } catch (e) {
    console.error(`  ERR ${filename}: ${e}`);
  }
}

async function main() {
  const currentHash = computeAlgorithmHash();
  console.log(`Algorithm hash: ${currentHash}`);

  const { rows: annRows } = await db.query('SELECT filename FROM annotations');
  const annotated: string[] = annRows.map((r: any) => r.filename as string);

  const { rows: genRows } = await db.query('SELECT filename, algorithm_hash FROM generated');
  const inGenerated = new Map<string, string>(genRows.map((r: any) => [r.filename, r.algorithm_hash as string]));

  const stale = annotated.filter(f => inGenerated.get(f) !== currentHash);
  console.log(`${annotated.length} annotated images, ${stale.length} stale/new`);

  if (stale.length === 0) { console.log('Nothing to do.'); await db.end(); return; }

  // Process in parallel with CONCURRENCY limit
  let idx = 0;
  async function worker() {
    while (idx < stale.length) {
      const filename = stale[idx++];
      await processImage(filename, currentHash);
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

  console.log('\nDone.');
  await db.end();
}

main().catch(e => { console.error(e); process.exit(1); });
