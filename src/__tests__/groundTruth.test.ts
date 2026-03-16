import * as path from 'path';
import * as fs from 'fs';
import { Pool } from 'pg';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../../images');

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

afterAll(() => db.end());

interface SplineRing {
  points: [number, number][];
}

interface ImageAnnotation {
  paperBoundary: [number, number][] | null;
  rings: SplineRing[];
}

function splineCentroid(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

function splineRadius(ring: SplineRing): number {
  const [cx, cy] = splineCentroid(ring);
  const radii = ring.points.map(([x, y]) => Math.hypot(x - cx, y - cy));
  return radii.reduce((s, r) => s + r, 0) / radii.length;
}

const imageFiles = fs
  .readdirSync(IMAGES_DIR)
  .filter(f => /\.(jpg|jpeg)$/i.test(f))
  .sort();

test.each(imageFiles)(
  'ground truth: %s',
  async (filename) => {
    const { rows } = await db.query(
      'SELECT paper_boundary, rings FROM annotations WHERE filename = $1',
      [filename],
    );
    expect(rows.length).toBeGreaterThan(0); // fail if no annotation in DB
    const ann: ImageAnnotation = { paperBoundary: rows[0].paper_boundary, rings: rows[0].rings };

    const imgPath = path.join(IMAGES_DIR, filename);
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);

    // Paper boundary: each annotated corner within 60px of the nearest detected vertex
    if (ann.paperBoundary && result.paperBoundary) {
      for (const annCorner of ann.paperBoundary) {
        const minDist = Math.min(
          ...result.paperBoundary.points.map(([x, y]) =>
            Math.hypot(x - annCorner[0], y - annCorner[1])
          ),
        );
        expect(minDist).toBeLessThan(60);
      }
    }

    // Center of ring[0] within 25px
    if (ann.rings.length > 0 && result.rings.length > 0) {
      const [annCx, annCy] = splineCentroid(ann.rings[0]);
      const { centerX, centerY } = result.rings[0];
      expect(Math.hypot(centerX - annCx, centerY - annCy)).toBeLessThan(25);
    }

    // Ring radius within 15% for each ring
    const minLen = Math.min(ann.rings.length, result.rings.length);
    for (let i = 0; i < minLen; i++) {
      const annRadius = splineRadius(ann.rings[i]);
      const resRadius = result.rings[i].width / 2;
      expect(Math.abs(resRadius - annRadius) / annRadius).toBeLessThan(0.15);
    }
  },
  120000,
);
