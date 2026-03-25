import {
  evalCatmullRom,
  sampleClosedSpline,
  pointInClosedSpline,
  ellipseToSplinePoints,
} from '../spline';

// ---------------------------------------------------------------------------
// evalCatmullRom
// ---------------------------------------------------------------------------

describe('evalCatmullRom', () => {
  // For a collinear set of equally-spaced points, Catmull-Rom is linear.
  // Points: (0,0), (1,0), (2,0), (3,0) — all on the x-axis, uniform spacing.
  const p0: [number, number] = [0, 0];
  const p1: [number, number] = [1, 0];
  const p2: [number, number] = [2, 0];
  const p3: [number, number] = [3, 0];

  it('t=0 returns p1', () => {
    const [x, y] = evalCatmullRom(p0, p1, p2, p3, 0);
    expect(x).toBeCloseTo(1);
    expect(y).toBeCloseTo(0);
  });

  it('t=1 returns p2', () => {
    const [x, y] = evalCatmullRom(p0, p1, p2, p3, 1);
    expect(x).toBeCloseTo(2);
    expect(y).toBeCloseTo(0);
  });

  it('t=0.5 interpolates linearly for uniform collinear points', () => {
    const [x, y] = evalCatmullRom(p0, p1, p2, p3, 0.5);
    expect(x).toBeCloseTo(1.5);
    expect(y).toBeCloseTo(0);
  });

  it('works in 2D with y variation', () => {
    const a: [number, number] = [0, 0];
    const b: [number, number] = [0, 1];
    const c: [number, number] = [0, 2];
    const d: [number, number] = [0, 3];
    const [x, y] = evalCatmullRom(a, b, c, d, 0);
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(1);
  });
});

// ---------------------------------------------------------------------------
// sampleClosedSpline
// ---------------------------------------------------------------------------

describe('sampleClosedSpline', () => {
  it('returns the single point for a 1-point ring', () => {
    const pts: [number, number][] = [[5, 7]];
    const result = sampleClosedSpline(pts, 8);
    expect(result).toEqual([[5, 7]]);
  });

  it('returns exactly nSamples points for a square ring', () => {
    // 4 control points → nSamples should be at least 4
    const square: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const result = sampleClosedSpline(square, 16);
    // sps = ceil(16/4) = 4; 4 segments × 4 = 16 points
    expect(result).toHaveLength(16);
  });

  it('produces points that stay near the control points for a circle ring', () => {
    // 8 control points on a unit circle
    const K = 8;
    const circle: [number, number][] = Array.from({ length: K }, (_, i) => {
      const a = (2 * Math.PI * i) / K;
      return [Math.cos(a), Math.sin(a)];
    });
    const result = sampleClosedSpline(circle, 64);
    for (const [x, y] of result) {
      expect(Math.hypot(x, y)).toBeCloseTo(1, 1); // within ~0.1 of unit radius
    }
  });
});

// ---------------------------------------------------------------------------
// pointInClosedSpline
// ---------------------------------------------------------------------------

describe('pointInClosedSpline', () => {
  // Use a large square so the spline approximation stays close to the edges.
  const square: [number, number][] = [
    [0, 0], [100, 0], [100, 100], [0, 100],
  ];

  it('returns true for a point clearly inside', () => {
    expect(pointInClosedSpline([50, 50], square)).toBe(true);
  });

  it('returns false for a point clearly outside', () => {
    expect(pointInClosedSpline([200, 200], square)).toBe(false);
  });

  it('returns false for a point far to the left', () => {
    expect(pointInClosedSpline([-50, 50], square)).toBe(false);
  });

  it('returns true for a point near the centre of a circle ring', () => {
    const K = 12;
    const r = 100;
    const circle: [number, number][] = Array.from({ length: K }, (_, i) => {
      const a = (2 * Math.PI * i) / K;
      return [r * Math.cos(a), r * Math.sin(a)];
    });
    expect(pointInClosedSpline([0, 0], circle)).toBe(true);
    expect(pointInClosedSpline([0, 150], circle)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// ellipseToSplinePoints
// ---------------------------------------------------------------------------

describe('ellipseToSplinePoints', () => {
  it('generates K control points (default K=12)', () => {
    const pts = ellipseToSplinePoints(0, 0, 50, 30, 0);
    expect(pts).toHaveLength(12);
  });

  it('generates K control points when K is specified', () => {
    const pts = ellipseToSplinePoints(0, 0, 50, 30, 0, 8);
    expect(pts).toHaveLength(8);
  });

  it('all points lie on the ellipse for angle=0', () => {
    const cx = 100, cy = 200, rx = 60, ry = 40;
    const pts = ellipseToSplinePoints(cx, cy, rx, ry, 0, 16);
    for (const [x, y] of pts) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      expect(dx * dx + dy * dy).toBeCloseTo(1, 5);
    }
  });

  it('centre point is the mean of all generated points', () => {
    const cx = 50, cy = 80;
    const pts = ellipseToSplinePoints(cx, cy, 40, 25, 30);
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    expect(mx).toBeCloseTo(cx, 5);
    expect(my).toBeCloseTo(cy, 5);
  });

  it('rotation by 90° swaps rx/ry axes', () => {
    // rx along x-axis → after 90° rotation, the widest extent should be along y
    const pts0 = ellipseToSplinePoints(0, 0, 50, 10, 0);
    const pts90 = ellipseToSplinePoints(0, 0, 50, 10, 90);
    const xExtent0 = Math.max(...pts0.map(p => Math.abs(p[0])));
    const xExtent90 = Math.max(...pts90.map(p => Math.abs(p[0])));
    // After 90° rotation the x extent should shrink significantly
    expect(xExtent90).toBeLessThan(xExtent0 * 0.5);
  });
});
