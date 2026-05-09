/**
 * Migration: remove out-of-image-bounds control points from annotations and
 * generated tables.
 *
 * For each row:
 *  - load the source image to get width × height
 *  - filter paper_boundary and ring points to [0,w] × [0,h]
 *  - drop rings with < 3 remaining points
 *  - set paper_boundary to null if < 3 points remain
 *  - update the row only if data changed
 *
 * Run with:
 *   npx tsx scripts/migrate-clamp-points.ts
 */
import * as path from 'path';
import * as fs from 'fs';
import { Pool } from 'pg';
import { loadImageNode } from '../src/imageLoader';

const IMAGES_DIR = path.resolve(__dirname, '../images');

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

type Point = [number, number];
interface SplineRing { points: Point[]; }

function clamp(
  rings: SplineRing[],
  boundary: Point[] | null,
  width: number,
  height: number,
): { rings: SplineRing[]; boundary: Point[] | null; changed: boolean } {
  const inBounds = ([x, y]: Point) => x >= 0 && x <= width && y >= 0 && y <= height;

  const newRings = rings
    .map(r => ({ points: r.points.filter(inBounds) }))
    .filter(r => r.points.length >= 3);

  const newBoundary = boundary
    ? (() => { const b = boundary.filter(inBounds); return b.length >= 3 ? b : null; })()
    : null;

  const changed =
    newRings.length !== rings.length ||
    newRings.some((r, i) => r.points.length !== rings[i].points.length) ||
    (boundary === null) !== (newBoundary === null) ||
    (boundary !== null && newBoundary !== null && newBoundary.length !== boundary.length);

  return { rings: newRings, boundary: newBoundary, changed };
}

async function getImageSize(filename: string): Promise<{ width: number; height: number } | null> {
  const imgPath = path.join(IMAGES_DIR, filename);
  if (!fs.existsSync(imgPath)) return null;
  const { width, height } = await loadImageNode(imgPath);
  return { width, height };
}

async function migrateTable(table: 'annotations' | 'generated') {
  const { rows } = await db.query(
    `SELECT filename, paper_boundary, rings FROM ${table}`,
  );
  console.log(`\n[${table}] ${rows.length} row(s)`);

  let updated = 0, skipped = 0, missing = 0;

  for (const row of rows) {
    const size = await getImageSize(row.filename);
    if (!size) {
      console.log(`  SKIP ${row.filename} — image file not found`);
      missing++;
      continue;
    }

    const rings: SplineRing[] = row.rings ?? [];
    const boundary: Point[] | null = row.paper_boundary ?? null;
    const { rings: newRings, boundary: newBoundary, changed } = clamp(
      rings, boundary, size.width, size.height,
    );

    if (!changed) { skipped++; continue; }

    await db.query(
      `UPDATE ${table} SET rings = $1, paper_boundary = $2 WHERE filename = $3`,
      [JSON.stringify(newRings), newBoundary ? JSON.stringify(newBoundary) : null, row.filename],
    );
    console.log(
      `  UPDATE ${row.filename}` +
      (rings.length !== newRings.length ? ` rings ${rings.length}→${newRings.length}` : '') +
      (boundary !== null && boundary.length !== (newBoundary?.length ?? 0)
        ? ` boundary ${boundary.length}→${newBoundary?.length ?? 'null'}`
        : ''),
    );
    updated++;
  }

  console.log(`  done: ${updated} updated, ${skipped} unchanged, ${missing} missing image`);
}

async function main() {
  await migrateTable('annotations');
  await migrateTable('generated');
  await db.end();
}

main().catch(err => { console.error(err); process.exit(1); });
