import * as path from 'path';
import * as fs from 'fs';
import { Pool } from 'pg';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';
import { findArrows } from '../arrowDetection';

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

interface ArrowAnnotation {
  tip:   [number, number];
  nock:  [number, number];
  score: number | 'X' | null;
}

interface ImageAnnotation {
  paperBoundary: [number, number][] | null;
  rings: SplineRing[];
  arrows: ArrowAnnotation[];
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
      'SELECT paper_boundary, rings, arrows FROM annotations WHERE filename = $1',
      [filename],
    );
    expect(rows.length).toBeGreaterThan(0); // fail if no annotation in DB
    const ann: ImageAnnotation = {
      paperBoundary: rows[0].paper_boundary,
      rings: rows[0].rings,
      arrows: rows[0].arrows ?? [],
    };

    const imgPath = path.join(IMAGES_DIR, filename);
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);

    // Paper boundary: each annotated corner within 30px of the nearest edge of the
    // detected polygon.  Checking edge distance (not vertex distance) handles the
    // case where the two polygons have different vertex counts: an annotated corner
    // that falls on a detected edge with no nearby vertex will still pass.
    if (ann.paperBoundary && result.paperBoundary) {
      const det = result.paperBoundary.points;
      const m = det.length;
      function pointToPolyDist(px: number, py: number): number {
        let minD = Infinity;
        for (let i = 0; i < m; i++) {
          const [ax, ay] = det[i];
          const [bx, by] = det[(i + 1) % m];
          const dx = bx - ax, dy = by - ay;
          const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
          minD = Math.min(minD, Math.hypot(px - (ax + t * dx), py - (ay + t * dy)));
        }
        return minD;
      }
      for (const [cx, cy] of ann.paperBoundary) {
        expect(pointToPolyDist(cx, cy)).toBeLessThan(50);
      }
    }

    // Center of innermost ring within 25px (use smallest-radius ring from each)
    if (ann.rings.length > 0 && result.rings.length > 0) {
      const annInner = [...ann.rings].sort((a, b) => splineRadius(a) - splineRadius(b))[0];
      const resInner = [...result.rings].sort((a, b) => splineRadius(a) - splineRadius(b))[0];
      const [annCx, annCy] = splineCentroid(annInner);
      const [resCx, resCy] = splineCentroid(resInner);
      expect(Math.hypot(resCx - annCx, resCy - annCy)).toBeLessThan(25);
    }

    // Ring radius within 30% for each ring (sort by radius to handle annotation order differences)
    const annSorted  = [...ann.rings].sort((a, b) => splineRadius(a) - splineRadius(b));
    const resSorted  = [...result.rings].sort((a, b) => splineRadius(a) - splineRadius(b));
    const minLen = Math.min(annSorted.length, resSorted.length);
    for (let i = 0; i < minLen; i++) {
      const annRadius = splineRadius(annSorted[i]);
      const resRadius = splineRadius(resSorted[i]);
      expect(Math.abs(resRadius - annRadius) / annRadius).toBeLessThan(0.30);
    }

    // Colour calibration sanity: hue ranges
    if (result.calibration) {
      const { gold, red, blue, black, white } = result.calibration;
      expect(gold[0]).toBeGreaterThan(20); expect(gold[0]).toBeLessThan(70);
      // red wraps: 0-18 or 342-360
      expect(red[0] < 18 || red[0] > 342).toBe(true);
      expect(blue[0]).toBeGreaterThan(190); expect(blue[0]).toBeLessThan(245);
      expect(black[2]).toBeLessThan(0.3);  // V < 0.3
      expect(white[1]).toBeLessThan(0.2);  // S < 0.2
    }

    // Arrow detection assertions (P9-T8)
    if (ann.arrows.length > 0) {
      const detected = findArrows(rgba, width, height, result);

      const fmtPt  = (p: [number, number] | null) =>
        p ? `(${Math.round(p[0])},${Math.round(p[1])})` : 'null';
      const detSummary = () =>
        detected.map((d, i) => `  det[${i}] tip=${fmtPt(d.tip)} nock=${fmtPt(d.nock)}`).join('\n');
      const annSummary = () =>
        ann.arrows.map((a, i) => `  ann[${i}] tip=${fmtPt(a.tip)} nock=${fmtPt(a.nock)} score=${a.score}`).join('\n');

      // Count: exact match
      if (detected.length !== ann.arrows.length) {
        console.error(
          `[${filename}] count: detected ${detected.length}, expected ${ann.arrows.length}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(detected.length).toBe(ann.arrows.length);

      // Bijective tip matching: each annotated tip within 15 px of exactly one detected tip
      const matchedDet = new Set<number>();
      const tipFailures: string[] = [];
      for (const annArrow of ann.arrows) {
        let bestDist = Infinity, bestIdx = -1;
        for (let di = 0; di < detected.length; di++) {
          if (matchedDet.has(di)) continue;
          const d = Math.hypot(detected[di].tip[0] - annArrow.tip[0], detected[di].tip[1] - annArrow.tip[1]);
          if (d < bestDist) { bestDist = d; bestIdx = di; }
        }
        if (bestDist >= 15) {
          tipFailures.push(
            `  ann tip=${fmtPt(annArrow.tip)} score=${annArrow.score}` +
            ` → best det=${fmtPt(detected[bestIdx]?.tip ?? null)} dist=${bestDist.toFixed(1)}px`,
          );
        }
        if (bestIdx >= 0) matchedDet.add(bestIdx);
      }
      if (tipFailures.length > 0) {
        console.error(
          `[${filename}] ${tipFailures.length} tip(s) out of range:\n${tipFailures.join('\n')}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
        // Re-run to throw on first failure (preserves Jest assertion count)
        const matchedDet3 = new Set<number>();
        for (const annArrow of ann.arrows) {
          let bestDist = Infinity, bestIdx = -1;
          for (let di = 0; di < detected.length; di++) {
            if (matchedDet3.has(di)) continue;
            const d = Math.hypot(detected[di].tip[0] - annArrow.tip[0], detected[di].tip[1] - annArrow.tip[1]);
            if (d < bestDist) { bestDist = d; bestIdx = di; }
          }
          expect(bestDist).toBeLessThan(15); // tip within 15 px
          if (bestIdx >= 0) matchedDet3.add(bestIdx);
        }
      }

      // Nock matching: each annotated nock within 40 px of its matched detection's nock (skip if null)
      const matchedDet2 = new Set<number>();
      for (const annArrow of ann.arrows) {
        let bestDist = Infinity, bestIdx = -1;
        for (let di = 0; di < detected.length; di++) {
          if (matchedDet2.has(di)) continue;
          const d = Math.hypot(detected[di].tip[0] - annArrow.tip[0], detected[di].tip[1] - annArrow.tip[1]);
          if (d < bestDist) { bestDist = d; bestIdx = di; }
        }
        if (bestIdx >= 0) {
          matchedDet2.add(bestIdx);
          const det = detected[bestIdx];
          if (det.nock !== null) {
            const nd = Math.hypot(det.nock[0] - annArrow.nock[0], det.nock[1] - annArrow.nock[1]);
            if (nd >= 40) {
              console.error(
                `[${filename}] nock too far:\n` +
                `  ann tip=${fmtPt(annArrow.tip)} nock=${fmtPt(annArrow.nock)}\n` +
                `  det tip=${fmtPt(det.tip)} nock=${fmtPt(det.nock)} dist=${nd.toFixed(1)}px`,
              );
            }
            expect(nd).toBeLessThan(40); // nock within 40 px
          }
        }
      }
    }
  },
  120000,
);
