import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../../images');

const jpgFiles = fs.readdirSync(IMAGES_DIR)
  .filter(f => f.endsWith('.jpg'))
  .map(f => path.join(IMAGES_DIR, f));

describe('findTarget', () => {
  test.each(jpgFiles)('%s — detects 10 concentric rings', async (imgPath) => {
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);
    expect(result.rings).toHaveLength(10);

    // --- Basic sanity ---
    for (const ring of result.rings) {
      expect(ring.width).toBeGreaterThan(0);
      expect(ring.height).toBeGreaterThan(0);
    }

    // Target centre must be well inside the image (at least 5% margin from every edge)
    const margin = 0.05;
    expect(result.rings[0].centerX).toBeGreaterThan(width  * margin);
    expect(result.rings[0].centerX).toBeLessThan(width  * (1 - margin));
    expect(result.rings[0].centerY).toBeGreaterThan(height * margin);
    expect(result.rings[0].centerY).toBeLessThan(height * (1 - margin));

    // All centers coincide (forced concentric) — within 2 px of ring[0]
    const { centerX: ox, centerY: oy } = result.rings[0];
    for (const ring of result.rings) {
      expect(Math.hypot(ring.centerX - ox, ring.centerY - oy)).toBeLessThanOrEqual(2);
    }

    // --- Ring size / scale ---
    const shortSide = Math.min(width, height);

    // Bullseye must be at least 5% of the short image side
    expect(result.rings[0].width).toBeGreaterThan(shortSide * 0.05);

    // Outermost ring: must be visible (≥ 20% of short side) and not overblown (< 150%)
    expect(result.rings[9].width).toBeGreaterThan(shortSide * 0.20);
    expect(result.rings[9].width).toBeLessThan(shortSide * 1.50);

    // --- Monotone growth (both axes, strictly) ---
    for (let i = 0; i < result.rings.length - 1; i++) {
      expect(result.rings[i].width).toBeLessThan(result.rings[i + 1].width);
      expect(result.rings[i].height).toBeLessThan(result.rings[i + 1].height);
    }

    // Consecutive width ratio must be meaningful: each ring at least 5% wider
    // than its predecessor (theoretical WA minimum is ~1.11 for the outer pair).
    // Upper bound 3× prevents runaway extrapolation artefacts.
    for (let i = 0; i < result.rings.length - 1; i++) {
      const ratio = result.rings[i + 1].width / result.rings[i].width;
      expect(ratio).toBeGreaterThan(1.05);
      expect(ratio).toBeLessThan(3.0);
    }

    // --- Aspect ratio ---
    const aspectRatios = result.rings.map(r => r.width / r.height);

    // No ring should be extremely elongated (steep angles still < 3:1)
    for (const ar of aspectRatios) {
      expect(ar).toBeLessThan(3);
    }

    // All rings are projections of concentric circles → they share the same AR.
    // Allow up to 0.8 spread to tolerate imperfect Fitzgibbon fits on partial arcs.
    const arSpread = Math.max(...aspectRatios) - Math.min(...aspectRatios);
    expect(arSpread).toBeLessThan(0.8);

    // --- Boundary containment ---
    // Use polygon inradius (min distance from centre to each polygon edge) as
    // a tight bound.  All rings except the outermost (ring[9]) must fit inside;
    // ring[9] may touch or slightly exceed the inradius (it IS the boundary),
    // but must stay within the max vertex distance with a small 15% slack.
    if (result.paperBoundary && result.paperBoundary.points.length >= 3) {
      const cx = result.rings[0].centerX, cy = result.rings[0].centerY;
      const pts = result.paperBoundary.points;
      const n = pts.length;

      function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
        const dx = bx - ax, dy = by - ay;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
      }

      const inradius = Math.min(...pts.map(([ax, ay], i) => {
        const [bx, by] = pts[(i + 1) % n];
        return distToSegment(cx, cy, ax, ay, bx, by);
      }));
      const maxVertexR = Math.max(...pts.map(([bx, by]) => Math.hypot(bx - cx, by - cy)));

      // Rings 0-8: must stay within inradius (the actual paper boundary)
      for (const ring of result.rings.slice(0, 9)) {
        expect(ring.width / 2).toBeLessThan(inradius * 1.15);
      }
      // Ring 9: outermost, allowed up to max vertex distance + 15% slack
      expect(result.rings[9].width / 2).toBeLessThan(maxVertexR * 1.15);
    }
  }, 60000);
});
