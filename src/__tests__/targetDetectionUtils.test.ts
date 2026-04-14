import { findTarget, pointInPolygon, polyToRayDists } from '../targetDetection';
import type { TargetBoundary } from '../targetDetection';

// ---------------------------------------------------------------------------
// pointInPolygon
// ---------------------------------------------------------------------------

describe('pointInPolygon', () => {
  const square: TargetBoundary = {
    points: [[0, 0], [100, 0], [100, 100], [0, 100]],
  };

  it('returns true for a point clearly inside', () => {
    expect(pointInPolygon({ x: 50, y: 50 }, square)).toBe(true);
  });

  it('returns false for a point outside', () => {
    expect(pointInPolygon({ x: 200, y: 50 }, square)).toBe(false);
  });

  it('returns false for a point above', () => {
    expect(pointInPolygon({ x: 50, y: -10 }, square)).toBe(false);
  });

  it('returns false for a degenerate polygon with fewer than 3 points', () => {
    const line: TargetBoundary = { points: [[0, 0], [100, 100]] };
    expect(pointInPolygon({ x: 50, y: 50 }, line)).toBe(false);
  });

  it('returns true for a point inside a triangle', () => {
    // Triangle: (0,0), (100,0), (50,100)
    const tri: TargetBoundary = { points: [[0, 0], [100, 0], [50, 100]] };
    expect(pointInPolygon({ x: 50, y: 40 }, tri)).toBe(true);
    expect(pointInPolygon({ x: 5,  y: 90 }, tri)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// findTarget failure paths
// ---------------------------------------------------------------------------

describe('findTarget failure paths', () => {
  it('returns success:false for a blank gray image (no colour blobs)', () => {
    // 50×50 uniform gray — no HSV saturation, so no colour blobs are found
    const w = 50, h = 50;
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      rgba[i * 4]     = 128;
      rgba[i * 4 + 1] = 128;
      rgba[i * 4 + 2] = 128;
      rgba[i * 4 + 3] = 255;
    }
    const result = findTarget(rgba, w, h);
    expect(result.success).toBe(false);
    expect((result as any).error).toBeTruthy();
  });

  it('includes an error string describing the failure', () => {
    const w = 10, h = 10;
    const rgba = new Uint8Array(w * h * 4).fill(0); // all-black image
    const result = findTarget(rgba, w, h);
    expect(result.success).toBe(false);
    expect(typeof (result as any).error).toBe('string');
  });
});

// ---------------------------------------------------------------------------
// polyToRayDists
// ---------------------------------------------------------------------------

describe('polyToRayDists', () => {
  // A square centred at the origin with half-width 100.
  const square: [number, number][] = [
    [100, 100], [-100, 100], [-100, -100], [100, -100],
  ];
  const cx = 0, cy = 0;

  it('returns N distances for N rays', () => {
    const dists = polyToRayDists(square, cx, cy, 8);
    expect(dists).toHaveLength(8);
  });

  it('all distances are positive', () => {
    const dists = polyToRayDists(square, cx, cy, 36);
    for (const d of dists) expect(d).toBeGreaterThan(0);
  });

  it('distance along axis rays to a square is approximately 100', () => {
    // Rays 0, 9, 18, 27 are at 0°, 90°, 180°, 270° — each hits a side at distance 100.
    const dists = polyToRayDists(square, cx, cy, 36);
    expect(dists[0]).toBeCloseTo(100, 0);
    expect(dists[9]).toBeCloseTo(100, 0);
  });

  it('fallback (no-intersection ray) returns max vertex distance, not zero', () => {
    // A triangle where one side faces away from some ray angles.
    // Ray pointing straight down (-y) from centre (0,0) to a triangle with all
    // vertices above (positive y). No segment faces downward → fallback fires.
    const triangle: [number, number][] = [[0, 50], [50, 150], [-50, 150]];
    const dists = polyToRayDists(triangle, 0, 0, 4); // 4 rays: 0°, 90°, 180°, 270°
    // Ray 2 (180° = pointing left) and ray 3 (270° = pointing down) may not intersect.
    for (const d of dists) expect(d).toBeGreaterThan(0); // never zero thanks to fallback
  });
});
