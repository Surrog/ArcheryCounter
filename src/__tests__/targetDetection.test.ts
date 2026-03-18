import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';
import type { SplineRing } from '../spline';

const IMAGES_DIR = path.resolve(__dirname, '../../images');

const jpgFiles = fs.readdirSync(IMAGES_DIR)
  .filter(f => f.endsWith('.jpg'))
  .map(f => path.join(IMAGES_DIR, f));

/** Mean of the control-point coordinates — approximates the ring centre. */
function splineCentroid(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

/** Mean radial distance from the centroid — approximates ring radius. */
function splineRadius(ring: SplineRing): number {
  const [cx, cy] = splineCentroid(ring);
  const radii = ring.points.map(([x, y]) => Math.hypot(x - cx, y - cy));
  return radii.reduce((s, r) => s + r, 0) / radii.length;
}

/**
 * Approximate semi-axis widths: compute projections of all control points along
 * the horizontal and vertical axes from the centroid.
 */
function splineAxes(ring: SplineRing): { rx: number; ry: number } {
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

    // Bullseye must be at least 2.5% of the short image side
    expect(r0).toBeGreaterThan(shortSide * 0.025);

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
