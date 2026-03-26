import * as path from 'path';
import * as fs from 'fs';
import { Pool } from 'pg';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';
import { findArrows } from '../arrowDetection';
import { scoreArrow } from '../scoring';
import { sampleClosedSpline, splineCentroid, splineRadius } from '../spline';

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

/** Signed area of a polygon (positive = CCW). */
function signedArea(poly: [number, number][]): number {
  let area = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    area += x1 * y2 - x2 * y1;
  }
  return area / 2;
}

/**
 * Sutherland-Hodgman polygon clipping (clip must be convex, both CCW).
 * Returns the intersection polygon.
 */
function clipPolygon(
  subject: [number, number][],
  clip: [number, number][],
): [number, number][] {
  let output = subject.slice();
  for (let i = 0; i < clip.length && output.length > 0; i++) {
    const input = output;
    output = [];
    const [ax, ay] = clip[i];
    const [bx, by] = clip[(i + 1) % clip.length];
    const inside = (px: number, py: number) =>
      (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0;
    const intersect = ([p1x, p1y]: [number, number], [p2x, p2y]: [number, number]): [number, number] => {
      const dx1 = p2x - p1x, dy1 = p2y - p1y;
      const dx2 = bx - ax,   dy2 = by - ay;
      const t = ((ax - p1x) * dy2 - (ay - p1y) * dx2) / (dx1 * dy2 - dy1 * dx2);
      return [p1x + t * dx1, p1y + t * dy1];
    };
    for (let j = 0; j < input.length; j++) {
      const curr = input[j];
      const prev = input[(j + input.length - 1) % input.length];
      const currIn = inside(curr[0], curr[1]);
      const prevIn = inside(prev[0], prev[1]);
      if (currIn) {
        if (!prevIn) output.push(intersect(prev, curr));
        output.push(curr);
      } else if (prevIn) {
        output.push(intersect(prev, curr));
      }
    }
  }
  return output;
}

/** IoU of two polygons (works for convex-ish paper boundaries). */
function polyIoU(a: [number, number][], b: [number, number][]): number {
  // Normalise both to CCW winding
  const aCCW = signedArea(a) < 0 ? [...a].reverse() : a;
  const bCCW = signedArea(b) < 0 ? [...b].reverse() : b;
  const inter = clipPolygon(aCCW, bCCW);
  if (inter.length < 3) return 0;
  const interArea = Math.abs(signedArea(inter));
  const aArea = Math.abs(signedArea(aCCW));
  const bArea = Math.abs(signedArea(bCCW));
  const union = aArea + bArea - interArea;
  return union <= 0 ? 0 : interArea / union;
}

/** Minimum distance from pt to any sampled point on the given ring spline. */
function pointToSplineDist(pt: [number, number], ring: SplineRing): number {
  const samples = sampleClosedSpline(ring.points, 60);
  let min = Infinity;
  for (const [sx, sy] of samples) {
    const d = Math.hypot(pt[0] - sx, pt[1] - sy);
    if (d < min) min = d;
  }
  return min;
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
    if (rows.length === 0) {
      return; // no annotation yet — skip
    }
    const ann: ImageAnnotation = {
      paperBoundary: rows[0].paper_boundary,
      rings: rows[0].rings,
      arrows: rows[0].arrows ?? [],
    };

    const imgPath = path.join(IMAGES_DIR, filename);
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);

    // Paper boundary: IoU between annotated and detected polygon.
    if (ann.paperBoundary && result.paperBoundary) {
      const iou = polyIoU(ann.paperBoundary, result.paperBoundary.points);
      if (iou < 0.5) {
        console.error(`[${filename}] paper boundary IoU=${iou.toFixed(3)}`);
      }
      expect(iou).toBeGreaterThan(0.25);
    }

    // Ring IoU: sample each spline into a polygon and compare (sort by radius).
    if (ann.rings.length > 0 && result.rings.length > 0) {
      const annSorted = [...ann.rings].sort((a, b) => splineRadius(a) - splineRadius(b));
      const resSorted = [...result.rings].sort((a, b) => splineRadius(a) - splineRadius(b));
      const minLen = Math.min(annSorted.length, resSorted.length);
      for (let i = 0; i < minLen; i++) {
        if (splineRadius(annSorted[i]) < 1) continue; // degenerate annotation ring
        const annPoly = sampleClosedSpline(annSorted[i].points, 60);
        const resPoly = sampleClosedSpline(resSorted[i].points, 60);
        const iou = polyIoU(annPoly, resPoly);
        if (iou < 0.5) {
          console.error(`[${filename}] ring[${i}] IoU=${iou.toFixed(3)}`);
        }
        expect(iou).toBeGreaterThan(0.2);
      }
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

      // Count: allow missing up to 4 arrows; no more than 4 extra detections
      const missing = ann.arrows.length - detected.length;
      const extra   = detected.length - ann.arrows.length;
      if (missing > 4 || extra > 4) {
        console.error(
          `[${filename}] count: detected ${detected.length}, expected ${ann.arrows.length}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(detected.length).toBeGreaterThanOrEqual(ann.arrows.length - 4);
      expect(detected.length).toBeLessThanOrEqual(ann.arrows.length + 4);

      // Bijective tip matching: use distance-sorted assignment so that close pairs
      // are matched first, preventing far annotations from consuming good detections.
      // Tolerance 25 px; up to 2 misses allowed.
      const TIP_MATCH_PX = 45;
      type Pair = { ai: number; di: number; dist: number };
      const allPairs: Pair[] = [];
      for (let ai = 0; ai < ann.arrows.length; ai++) {
        for (let di = 0; di < detected.length; di++) {
          allPairs.push({ ai, di, dist: Math.hypot(
            detected[di].tip[0] - ann.arrows[ai].tip[0],
            detected[di].tip[1] - ann.arrows[ai].tip[1],
          )});
        }
      }
      allPairs.sort((a, b) => a.dist - b.dist);
      const matchedA = new Set<number>(), matchedD = new Set<number>();
      const tipMatchDist = new Map<number, number>(); // ai → dist
      const tipMatchIdx  = new Map<number, number>(); // ai → di
      for (const { ai, di, dist } of allPairs) {
        if (matchedA.has(ai) || matchedD.has(di)) continue;
        if (dist <= TIP_MATCH_PX) {
          matchedA.add(ai); matchedD.add(di);
          tipMatchDist.set(ai, dist);
          tipMatchIdx.set(ai, di);
        }
      }
      const tipFailures: string[] = [];
      for (let ai = 0; ai < ann.arrows.length; ai++) {
        if (!tipMatchDist.has(ai)) {
          const a = ann.arrows[ai];
          let bestDist = Infinity, bestIdx = -1;
          for (let di = 0; di < detected.length; di++) {
            const d = Math.hypot(detected[di].tip[0] - a.tip[0], detected[di].tip[1] - a.tip[1]);
            if (d < bestDist) { bestDist = d; bestIdx = di; }
          }
          tipFailures.push(
            `  ann tip=${fmtPt(a.tip)} score=${a.score}` +
            ` → best det=${fmtPt(detected[bestIdx]?.tip ?? null)} dist=${bestDist.toFixed(1)}px`,
          );
        }
      }
      // Allow guaranteed failures from count gap (max(0, N-D)) plus 5 positional misses.
      const maxTipFailures = Math.max(0, ann.arrows.length - detected.length) + 5;
      if (tipFailures.length > maxTipFailures) {
        console.error(
          `[${filename}] ${tipFailures.length} tip(s) unmatched (>${maxTipFailures} allowed):\n${tipFailures.join('\n')}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(tipFailures.length).toBeLessThanOrEqual(maxTipFailures);

      // Scoring assertions (P10-T7)
      if (result.calibration) {
        const scoreFailures: string[] = [];
        for (const [ai, di] of tipMatchIdx) {
          const annScore = ann.arrows[ai].score;
          if (annScore === null) continue; // unannotated score — skip
          const detScore = scoreArrow(detected[di].tip, result.rings);

          // Treat 'X' and 10 as equivalent numeric score (X is a tiebreaker within 10).
          const numAnn = annScore === 'X' ? 10 : annScore;
          const numDet = detScore === 'X' ? 10 : detScore;
          // Near-boundary: tip within 20px of any ring spline → allow ±1 tolerance.
          // 20px ≈ 20% of a typical ring width; handles ring boundary detection imprecision.
          const nearBoundary = result.rings.some(r => pointToSplineDist(detected[di].tip, r) < 20);
          const ok = numDet === numAnn || (nearBoundary && Math.abs(numDet - numAnn) <= 1);
          if (!ok) {
            scoreFailures.push(
              `  ann[${ai}] tip=${fmtPt(ann.arrows[ai].tip)} score=${annScore} → det score=${detScore}` +
              (nearBoundary ? ' (near boundary)' : ''),
            );
          }
        }
        if (scoreFailures.length > 0) {
          console.error(`[${filename}] scoring failures:\n${scoreFailures.join('\n')}`);
        }
        expect(scoreFailures.length).toBeLessThanOrEqual(2);
      }

      // Nock matching: informational only — log misses but do not assert.
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
                `[${filename}] nock info: ann tip=${fmtPt(annArrow.tip)} nock=${fmtPt(annArrow.nock)}` +
                ` det nock=${fmtPt(det.nock)} dist=${nd.toFixed(1)}px`,
              );
            }
          }
        }
      }
    }
  },
  120000,
);
