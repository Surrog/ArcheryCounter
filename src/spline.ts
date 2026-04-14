/**
 * Catmull-Rom spline utilities shared by the annotation tool (Phase 5)
 * and the algorithm output (Phase 6).
 */

/** A closed ring boundary represented as Catmull-Rom control points. */
export interface SplineRing {
  points: [number, number][];
}

/** Evaluates one Catmull-Rom segment at parameter t ∈ [0, 1]. */
export function evalCatmullRom(
  p0: [number, number], p1: [number, number],
  p2: [number, number], p3: [number, number],
  t: number,
): [number, number] {
  const t2 = t * t, t3 = t2 * t;
  return [
    0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
    0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
  ];
}

/** Samples a closed Catmull-Rom spline at nSamples evenly-spaced parameters. */
export function sampleClosedSpline(
  points: [number, number][], nSamples: number,
): [number, number][] {
  const K = points.length;
  if (K === 0) return [];
  if (K === 1) return Array.from({ length: nSamples }, () => points[0] as [number, number]);
  const result: [number, number][] = [];
  const sps = Math.ceil(nSamples / K);
  for (let k = 0; k < K; k++) {
    const p0 = points[(k - 1 + K) % K];
    const p1 = points[k];
    const p2 = points[(k + 1) % K];
    const p3 = points[(k + 2) % K];
    for (let s = 0; s < sps; s++) {
      result.push(evalCatmullRom(p0, p1, p2, p3, s / sps));
    }
  }
  return result;
}

/** Point-in-polygon test using ray casting on a polygon approximation of the spline (N=60). */
export function pointInClosedSpline(
  pt: [number, number], points: [number, number][],
): boolean {
  const poly = sampleClosedSpline(points, 60);
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    if (
      (poly[i][1] > pt[1]) !== (poly[j][1] > pt[1]) &&
      pt[0] < (poly[j][0] - poly[i][0]) * (pt[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]
    ) {
      inside = !inside;
    }
  }
  return inside;
}

/** Returns the mean of all control-point coordinates. */
export function splineCentroid(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

/** Returns the mean distance from the centroid to each control point. */
export function splineRadius(ring: SplineRing): number {
  const [cx, cy] = splineCentroid(ring);
  return ring.points.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / ring.points.length;
}

/**
 * Samples a rotated ellipse at K evenly-spaced angles to produce K Catmull-Rom
 * control points.  Used to initialise spline rings from ellipse detection output.
 */
export function ellipseToSplinePoints(
  cx: number, cy: number,
  rx: number, ry: number,
  angleDeg: number,
  K = 12,
): [number, number][] {
  const a = (angleDeg * Math.PI) / 180;
  const pts: [number, number][] = [];
  for (let k = 0; k < K; k++) {
    const theta = (2 * Math.PI * k) / K;
    const x = rx * Math.cos(theta);
    const y = ry * Math.sin(theta);
    pts.push([
      cx + x * Math.cos(a) - y * Math.sin(a),
      cy + x * Math.sin(a) + y * Math.cos(a),
    ]);
  }
  return pts;
}
