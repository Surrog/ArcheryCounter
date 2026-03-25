import { findTarget, pointInPolygon } from '../targetDetection';
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
