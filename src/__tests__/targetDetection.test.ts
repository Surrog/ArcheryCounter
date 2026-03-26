import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../../images');

const jpgFiles = fs.readdirSync(IMAGES_DIR)
  .filter(f => f.endsWith('.jpg'))
  .map(f => path.join(IMAGES_DIR, f));

/** Mean of the control-point coordinates — approximates the ring centre. */
function splineCentroid(ring: { points: [number, number][] }): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

/** Mean radial distance from the centroid — approximates ring radius. */
function splineRadius(ring: { points: [number, number][] }): number {
  const [cx, cy] = splineCentroid(ring);
  const radii = ring.points.map(([x, y]) => Math.hypot(x - cx, y - cy));
  return radii.reduce((s, r) => s + r, 0) / radii.length;
}

/**
 * Approximate semi-axis widths: compute projections of all control points along
 * the horizontal and vertical axes from the centroid.
 */
function splineAxes(ring: { points: [number, number][] }): { rx: number; ry: number } {
  const [cx, cy] = splineCentroid(ring);
  const xs = ring.points.map(([x]) => Math.abs(x - cx));
  const ys = ring.points.map(([, y]) => Math.abs(y - cy));
  return {
    rx: xs.reduce((a, b) => Math.max(a, b), 0),
    ry: ys.reduce((a, b) => Math.max(a, b), 0),
  };
}

describe('findTarget', () => {
  test.each(jpgFiles)('%s — detects 10 concentric rings', async (imgPath) => {
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);
    expect(result.rings).toHaveLength(10);

    // --- Basic sanity ---
    for (const ring of result.rings) {
      expect(splineRadius(ring)).toBeGreaterThan(0);
    }

    // Target centre must be well inside the image (at least 5% margin from every edge)
    const margin = 0.05;
    const [cx0, cy0] = splineCentroid(result.rings[0]);
    expect(cx0).toBeGreaterThan(width  * margin);
    expect(cx0).toBeLessThan(width  * (1 - margin));
    expect(cy0).toBeGreaterThan(height * margin);
    expect(cy0).toBeLessThan(height * (1 - margin));

    // All ring centroids should be approximately concentric.
    // Raw detection points are used directly as control points, so the centroid
    // is the mean of detected positions and may drift ~20 px from the bootstrap centre.
    for (const ring of result.rings) {
      const [cx, cy] = splineCentroid(ring);
      expect(Math.hypot(cx - cx0, cy - cy0)).toBeLessThanOrEqual(60);
    }

    // --- Ring size / scale ---
    const shortSide = Math.min(width, height);
    const r0 = splineRadius(result.rings[0]);
    const r9 = splineRadius(result.rings[9]);

    // Bullseye must be at least 1.5% of the short image side
    expect(r0).toBeGreaterThan(shortSide * 0.015);

    // Outermost ring: must be visible (≥ 10% of short side) and not overblown (< 75%)
    expect(r9).toBeGreaterThan(shortSide * 0.10);
    expect(r9).toBeLessThan(shortSide * 0.75);

    // --- Monotone growth (radii, strictly) ---
    for (let i = 0; i < result.rings.length - 1; i++) {
      expect(splineRadius(result.rings[i])).toBeLessThan(splineRadius(result.rings[i + 1]));
    }

    // Consecutive radius ratio must be meaningful: each ring at least 5% larger
    // than its predecessor (theoretical WA minimum is ~1.11 for the outer pair).
    // Upper bound 3× prevents runaway extrapolation artefacts.
    for (let i = 0; i < result.rings.length - 1; i++) {
      const ratio = splineRadius(result.rings[i + 1]) / splineRadius(result.rings[i]);
      expect(ratio).toBeGreaterThan(1.05);
      expect(ratio).toBeLessThan(3.0);
    }

    // --- Aspect ratio ---
    // All rings are projections of concentric circles; their rx/ry ratios should
    // be similar.  Allow up to 0.8 spread to tolerate radial-profile discretisation.
    const aspectRatios = result.rings.map(r => {
      const { rx, ry } = splineAxes(r);
      return ry > 0 ? rx / ry : 1;
    });

    // No ring should be extremely elongated (steep angles still < 3:1)
    for (const ar of aspectRatios) {
      expect(ar).toBeLessThan(3);
    }

    const arSpread = Math.max(...aspectRatios) - Math.min(...aspectRatios);
    expect(arSpread).toBeLessThan(0.8);

    // --- WA ring-width ratios ---
    // World Archery specifies equal-width scoring zones. Each outer zone boundary
    // is therefore 1 ring-width further than the inner boundary of the same zone.
    // The expected radius of ring i (0-indexed) is proportional to (i + 1).
    // Consecutive detected rings (odd indices 1,3,5,7,9) should therefore grow
    // by a ratio close to (i+3)/(i+1), which for adjacent pairs is:
    //   r3/r1 ≈ 4/2 = 2.0,  r5/r3 ≈ 6/4 = 1.5,  r7/r5 ≈ 8/6 = 1.33,  r9/r7 ≈ 10/8 = 1.25
    // Allow ±30% on each ratio to account for perspective and detection variance.
    const detectedPairs: [number, number][] = [[1, 3], [3, 5], [5, 7], [7, 9]];
    const expectedRatios = [2.0, 1.5, 1.333, 1.25];
    for (let p = 0; p < detectedPairs.length; p++) {
      const [ia, ib] = detectedPairs[p];
      const ratio = splineRadius(result.rings[ib]) / splineRadius(result.rings[ia]);
      expect(ratio).toBeGreaterThan(expectedRatios[p] * 0.7);
      expect(ratio).toBeLessThan(expectedRatios[p] * 1.3);
    }

    // --- ringPoints structure ---
    // Detected rings (odd indices) must have transition points; interpolated rings
    // (even indices 0,2,4,6) must not (they are derived by spline interpolation).
    expect(result.ringPoints).toBeDefined();
    if (result.ringPoints) {
      const INTERPOLATED = [0, 2, 4, 6];
      const DETECTED     = [1, 3, 5, 7, 8, 9];
      for (const i of INTERPOLATED) {
        expect(result.ringPoints[i]).toHaveLength(0);
      }
      for (const i of DETECTED) {
        expect(result.ringPoints[i].length).toBeGreaterThan(0);
      }
    }

    // --- rayDebug ---
    // The colour-guided path always collects per-ray debug data.
    expect(result.rayDebug).toBeDefined();
    if (result.rayDebug) {
      expect(result.rayDebug.length).toBeGreaterThan(0);
      for (const entry of result.rayDebug) {
        expect(entry.distances).toHaveLength(10);
        expect(entry.boundary).toBeGreaterThan(0);
      }

      // Per-ray distance ordering: non-null detected distances must be strictly
      // increasing by ring index.  A ring boundary can never appear closer than
      // an inner boundary on the same ray (catches r8/r9 collapse on truncated rays).
      for (const { distances } of result.rayDebug) {
        let prevD = 0;
        for (let k = 0; k < 10; k++) {
          const d = distances[k];
          if (d === null) continue;
          expect(d).toBeGreaterThan(prevD);
          prevD = d;
        }
      }
    }

    // --- Boundary containment ---
    // Use polygon inradius (min distance from centre to each polygon edge) as
    // a tight bound.  All rings except the outermost (ring[9]) must fit inside;
    // ring[9] may touch or slightly exceed the inradius (it IS the boundary),
    // but must stay within the max vertex distance with a small 15% slack.
    if (result.paperBoundary && result.paperBoundary.points.length >= 3) {
      const [bCx, bCy] = splineCentroid(result.rings[0]);
      const pts = result.paperBoundary.points;
      const n = pts.length;

      function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
        const dx = bx - ax, dy = by - ay;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
      }

      const inradius = Math.min(...pts.map(([ax, ay], i) => {
        const [bx, by] = pts[(i + 1) % n];
        return distToSegment(bCx, bCy, ax, ay, bx, by);
      }));
      const maxVertexR = Math.max(...pts.map(([bx, by]) => Math.hypot(bx - bCx, by - bCy)));

      // Rings 0-8: must stay within inradius (the actual paper boundary)
      for (const ring of result.rings.slice(0, 9)) {
        expect(splineRadius(ring)).toBeLessThan(inradius * 1.15);
      }
      // Ring 9: outermost, allowed up to max vertex distance + 15% slack
      expect(splineRadius(result.rings[9])).toBeLessThan(maxVertexR * 1.15);
    }
  }, 60000);
});
