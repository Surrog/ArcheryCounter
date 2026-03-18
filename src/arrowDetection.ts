/**
 * Phase 9 — Arrow detection pipeline.
 * See docs/plan.md §Phase 9 for design rationale.
 *
 * Pipeline:
 *   P9-T1  Hough segment extraction (downsampled 2×)
 *   P9-T2  Segment merging: centerline (edge-pair → midline) + collinear (crossing splits)
 *   P9-T3  Size + area + anti-ring filter
 *   P9-T4  Vane colour detection + match to shaft nock end
 *   P9-T5  Deduplication: cluster tips within 15 px
 *   P9-T6  Hole fallback (skipped in first iteration — wired but returns [])
 */

import type { ArcheryResult } from './targetDetection';

export interface ArrowDetection {
  tip:  [number, number];
  nock: [number, number] | null;
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------
type Pt  = [number, number];
type Seg = [Pt, Pt];

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

function ptInPoly(px: number, py: number, poly: Pt[]): boolean {
  let inside = false;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

function distToBoundary(px: number, py: number, poly: Pt[]): number {
  let minD = Infinity;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % n];
    const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
    minD = Math.min(minD, Math.hypot(px - (ax + t * dx), py - (ay + t * dy)));
  }
  return minD;
}

function polyBBox(poly: Pt[]) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const [x, y] of poly) {
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  }
  return { xMin, xMax, yMin, yMax };
}

/** Angle of segment normalised to [0, π). */
function segAngle(seg: Seg): number {
  let a = Math.atan2(seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]);
  if (a < 0) a += Math.PI;
  if (a >= Math.PI) a -= Math.PI;
  return a;
}

/** Unsigned angular difference, clamped to [0, π/2]. */
function angleDiff(a: number, b: number): number {
  let d = Math.abs(a - b);
  if (d > Math.PI / 2) d = Math.PI - d;
  return d;
}

function segLen(seg: Seg): number {
  return Math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]);
}

/** Perpendicular distance from point (px,py) to infinite line through (ax,ay)→(bx,by). */
function perpDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy);
  if (len < 1e-9) return Math.hypot(px - ax, py - ay);
  return Math.abs((py - ay) * dx - (px - ax) * dy) / len;
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const r1 = r / 255, g1 = g / 255, b1 = b / 255;
  const max = Math.max(r1, g1, b1), min = Math.min(r1, g1, b1);
  const d = max - min, v = max, s = max === 0 ? 0 : d / max;
  let h = 0;
  if (d > 0) {
    if      (max === r1) h = 60 * (((g1 - b1) / d) % 6);
    else if (max === g1) h = 60 * ((b1 - r1) / d + 2);
    else                 h = 60 * ((r1 - g1) / d + 4);
    if (h < 0) h += 360;
  }
  return [h, s, v];
}

// ---------------------------------------------------------------------------
// Step 1–2: Grayscale + downsample + Sobel
// ---------------------------------------------------------------------------

function buildGray(rgba: Uint8Array, w: number, h: number): Uint8Array {
  const g = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++)
    g[i] = Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]);
  return g;
}

function downsampleGray(
  gray: Uint8Array, width: number, height: number, scale: number,
): { g: Uint8Array; w: number; h: number } {
  const w = Math.floor(width / scale), h = Math.floor(height / scale);
  const g = new Uint8Array(w * h);
  const s2 = scale * scale;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let dy = 0; dy < scale; dy++)
        for (let dx = 0; dx < scale; dx++)
          sum += gray[(y * scale + dy) * width + (x * scale + dx)];
      g[y * w + x] = Math.round(sum / s2);
    }
  }
  return { g, w, h };
}

function sobelEdge(g: Uint8Array, w: number, h: number, thresh: number): Uint8Array {
  const e = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = (
        -g[(y-1)*w+(x-1)] + g[(y-1)*w+(x+1)]
        - 2*g[y*w+(x-1)] + 2*g[y*w+(x+1)]
        - g[(y+1)*w+(x-1)] + g[(y+1)*w+(x+1)]
      );
      const gy = (
        -g[(y-1)*w+(x-1)] - 2*g[(y-1)*w+x] - g[(y-1)*w+(x+1)]
        + g[(y+1)*w+(x-1)] + 2*g[(y+1)*w+x] + g[(y+1)*w+(x+1)]
      );
      e[y * w + x] = Math.hypot(gx, gy) > thresh ? 255 : 0;
    }
  }
  return e;
}

// ---------------------------------------------------------------------------
// P9-T1: Hough line segment extraction
// ---------------------------------------------------------------------------

const N_THETA = 180;

function houghSegments(
  edges: Uint8Array, w: number, h: number,
  bboxXMin: number, bboxXMax: number, bboxYMin: number, bboxYMax: number,
  minVotes: number, minLen: number, maxGap: number, maxPeaks: number,
): Seg[] {
  const diag = Math.ceil(Math.hypot(w, h));
  const nR   = 2 * diag + 1;
  const acc  = new Int32Array(N_THETA * nR);

  const cosT = new Float64Array(N_THETA);
  const sinT = new Float64Array(N_THETA);
  for (let t = 0; t < N_THETA; t++) {
    cosT[t] = Math.cos(t * Math.PI / N_THETA);
    sinT[t] = Math.sin(t * Math.PI / N_THETA);
  }

  // Vote with edge pixels inside the expanded boundary bbox
  const x0 = Math.max(1, Math.floor(bboxXMin));
  const x1 = Math.min(w - 2, Math.ceil(bboxXMax));
  const y0 = Math.max(1, Math.floor(bboxYMin));
  const y1 = Math.min(h - 2, Math.ceil(bboxYMax));

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (!edges[y * w + x]) continue;
      for (let t = 0; t < N_THETA; t++)
        acc[t * nR + Math.round(x * cosT[t] + y * sinT[t]) + diag]++;
    }
  }

  // Find local maxima with non-max suppression (±3 theta, ±8 r)
  const peaks: { t: number; r: number; v: number }[] = [];
  for (let t = 0; t < N_THETA; t++) {
    for (let r = 0; r < nR; r++) {
      const v = acc[t * nR + r];
      if (v < minVotes) continue;
      let isMax = true;
      outer: for (let dt = -3; dt <= 3 && isMax; dt++) {
        for (let dr = -8; dr <= 8 && isMax; dr++) {
          if (dt === 0 && dr === 0) continue;
          const t2 = ((t + dt) % N_THETA + N_THETA) % N_THETA;
          const r2  = r + dr;
          if (r2 >= 0 && r2 < nR && acc[t2 * nR + r2] > v) isMax = false;
        }
      }
      if (isMax) peaks.push({ t, r: r - diag, v });
    }
  }
  peaks.sort((a, b) => b.v - a.v);

  // Extract segments by walking along each peak line
  const segments: Seg[] = [];
  for (const { t, r } of peaks.slice(0, maxPeaks)) {
    const ct = cosT[t], st = sinT[t];
    const dirX = -st, dirY = ct;
    const bx = r * ct, by = r * st;

    let segStart: number | null = null, segEnd = 0, gap = 0;
    for (let s = -diag; s <= diag; s++) {
      const px = Math.round(bx + s * dirX);
      const py = Math.round(by + s * dirY);
      const on = px >= 0 && px < w && py >= 0 && py < h && edges[py * w + px] > 0;
      if (on) {
        if (segStart === null) segStart = s;
        segEnd = s; gap = 0;
      } else if (segStart !== null) {
        if (++gap > maxGap) {
          if (segEnd - segStart >= minLen)
            segments.push([
              [Math.round(bx + segStart * dirX), Math.round(by + segStart * dirY)],
              [Math.round(bx + segEnd   * dirX), Math.round(by + segEnd   * dirY)],
            ]);
          segStart = null; gap = 0;
        }
      }
    }
    if (segStart !== null && segEnd - segStart >= minLen)
      segments.push([
        [Math.round(bx + segStart * dirX), Math.round(by + segStart * dirY)],
        [Math.round(bx + segEnd   * dirX), Math.round(by + segEnd   * dirY)],
      ]);
  }
  return segments;
}

// ---------------------------------------------------------------------------
// P9-T2a: Centerline merge — collapse LSD edge-pairs into shaft midlines
// ---------------------------------------------------------------------------

function mergeCenterlines(segs: Seg[], angleTolDeg: number, perpTolPx: number): Seg[] {
  const angleTol = angleTolDeg * Math.PI / 180;
  const used = new Uint8Array(segs.length);
  const out: Seg[] = [];

  for (let i = 0; i < segs.length; i++) {
    if (used[i]) continue;
    const a = segs[i];
    const angA = segAngle(a);
    const dirX = Math.cos(angA), dirY = Math.sin(angA);
    // Reference midpoint
    const mAx = (a[0][0] + a[1][0]) / 2;
    const mAy = (a[0][1] + a[1][1]) / 2;

    let sumMidPerpOff = 0; // perpendicular offset of B midpoints from A's line, for averaging
    let partnerCount = 0;
    const allEndpts: Pt[] = [a[0], a[1]];

    for (let j = i + 1; j < segs.length; j++) {
      if (used[j]) continue;
      const b = segs[j];
      if (angleDiff(angA, segAngle(b)) > angleTol) continue;
      const mBx = (b[0][0] + b[1][0]) / 2;
      const mBy = (b[0][1] + b[1][1]) / 2;
      if (perpDist(mBx, mBy, mAx, mAy, mAx + dirX, mAy + dirY) > perpTolPx) continue;
      // Also require endpoint proximity
      const epd = Math.min(
        perpDist(b[0][0], b[0][1], a[0][0], a[0][1], a[1][0], a[1][1]),
        perpDist(b[1][0], b[1][1], a[0][0], a[0][1], a[1][0], a[1][1]),
      );
      if (epd > perpTolPx) continue;
      used[j] = 1;
      // Perpendicular offset of B midpoint relative to A midpoint (normal direction)
      const normX = -dirY, normY = dirX;
      sumMidPerpOff += (mBx - mAx) * normX + (mBy - mAy) * normY;
      partnerCount++;
      allEndpts.push(b[0], b[1]);
    }

    // Merged midline: average perpendicular offset, union extent along direction
    const perpOff = partnerCount > 0 ? sumMidPerpOff / (partnerCount + 1) : 0;
    const normX = -dirY, normY = dirX;
    const baseMx = mAx + perpOff * normX;
    const baseMy = mAy + perpOff * normY;
    const projs = allEndpts.map(p => (p[0] - baseMx) * dirX + (p[1] - baseMy) * dirY);
    const tMin = Math.min(...projs), tMax = Math.max(...projs);

    used[i] = 1;
    out.push([
      [Math.round(baseMx + tMin * dirX), Math.round(baseMy + tMin * dirY)],
      [Math.round(baseMx + tMax * dirX), Math.round(baseMy + tMax * dirY)],
    ]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// P9-T2b: Collinear merge — reassemble shaft halves split at crossings
// ---------------------------------------------------------------------------

function mergeCollinear(segs: Seg[], angleTolDeg: number, perpTolPx: number, gapTolPx: number): Seg[] {
  const angleTol = angleTolDeg * Math.PI / 180;
  let current = segs.slice();
  // Up to 4 passes to handle chains
  for (let pass = 0; pass < 4; pass++) {
    const used = new Uint8Array(current.length);
    const out: Seg[] = [];
    let merged = false;
    for (let i = 0; i < current.length; i++) {
      if (used[i]) continue;
      let a = current[i];
      const angA = segAngle(a);
      const dirX = Math.cos(angA), dirY = Math.sin(angA);
      for (let j = i + 1; j < current.length; j++) {
        if (used[j]) continue;
        const b = current[j];
        if (angleDiff(angA, segAngle(b)) > angleTol) continue;
        // Must be nearly on the same infinite line
        const d0 = perpDist(b[0][0], b[0][1], a[0][0], a[0][1], a[1][0], a[1][1]);
        const d1 = perpDist(b[1][0], b[1][1], a[0][0], a[0][1], a[1][0], a[1][1]);
        if (Math.min(d0, d1) > perpTolPx) continue;
        // Gap between their extents along the direction
        const ref = a[0];
        const prA = a.map(p  => (p[0]  - ref[0]) * dirX + (p[1]  - ref[1]) * dirY);
        const prB = b.map(p  => (p[0]  - ref[0]) * dirX + (p[1]  - ref[1]) * dirY);
        const maxA = Math.max(...prA), minA = Math.min(...prA);
        const maxB = Math.max(...prB), minB = Math.min(...prB);
        const gap = Math.max(0, Math.max(minA, minB) - Math.min(maxA, maxB));
        if (gap > gapTolPx) continue;
        // Merge
        const tAll = [...prA, ...prB];
        const tMin = Math.min(...tAll), tMax = Math.max(...tAll);
        a = [
          [Math.round(ref[0] + tMin * dirX), Math.round(ref[1] + tMin * dirY)],
          [Math.round(ref[0] + tMax * dirX), Math.round(ref[1] + tMax * dirY)],
        ];
        used[j] = 1;
        merged = true;
      }
      used[i] = 1;
      out.push(a);
    }
    current = out;
    if (!merged) break;
  }
  return current;
}

// ---------------------------------------------------------------------------
// P9-T3: Size + area + anti-ring filter
// ---------------------------------------------------------------------------

function filterSegments(
  segs: Seg[],
  boundary: Pt[],
  ringRadii: number[], cx: number, cy: number,
  minLen: number, nearBoundaryTol: number,
): Seg[] {
  return segs.filter(seg => {
    if (segLen(seg) < minLen) return false;

    const mx = (seg[0][0] + seg[1][0]) / 2;
    const my = (seg[0][1] + seg[1][1]) / 2;

    const nearBoundary = (
      ptInPoly(mx, my, boundary) ||
      ptInPoly(seg[0][0], seg[0][1], boundary) ||
      ptInPoly(seg[1][0], seg[1][1], boundary) ||
      distToBoundary(mx, my, boundary) < nearBoundaryTol ||
      distToBoundary(seg[0][0], seg[0][1], boundary) < nearBoundaryTol ||
      distToBoundary(seg[1][0], seg[1][1], boundary) < nearBoundaryTol
    );
    if (!nearBoundary) return false;

    // Anti-ring: reject if midpoint lies on a ring radius AND is nearly tangent
    const dToCenter = Math.hypot(mx - cx, my - cy);
    const segAng = segAngle(seg);
    for (const rr of ringRadii) {
      if (Math.abs(dToCenter - rr) < 10) {
        const tangentAng = Math.atan2(my - cy, mx - cx) + Math.PI / 2;
        const tNorm = ((tangentAng % Math.PI) + Math.PI) % Math.PI;
        if (angleDiff(segAng, tNorm) < 15 * Math.PI / 180) return false;
      }
    }
    return true;
  });
}

// ---------------------------------------------------------------------------
// Assign tip (endpoint closest to target centre) / nock
// ---------------------------------------------------------------------------

function assignTipNock(
  segs: Seg[], cx: number, cy: number,
): { tip: Pt; nock: Pt | null }[] {
  return segs.map(seg => {
    const d0 = Math.hypot(seg[0][0] - cx, seg[0][1] - cy);
    const d1 = Math.hypot(seg[1][0] - cx, seg[1][1] - cy);
    return d0 <= d1
      ? { tip: seg[0], nock: seg[1] }
      : { tip: seg[1], nock: seg[0] };
  });
}

// ---------------------------------------------------------------------------
// P9-T4: Vane colour blob detection
// ---------------------------------------------------------------------------

interface VaneBlob { cx: number; cy: number; area: number }

function detectVanes(
  rgba: Uint8Array, width: number, height: number,
  boundary: Pt[],
): VaneBlob[] {
  const MIN_AREA = 15, MAX_AREA = 800;

  const bbox = polyBBox(boundary);
  const x0 = Math.max(0, Math.floor(bbox.xMin - 100));
  const x1 = Math.min(width  - 1, Math.ceil(bbox.xMax + 100));
  const y0 = Math.max(0, Math.floor(bbox.yMin - 100));
  const y1 = Math.min(height - 1, Math.ceil(bbox.yMax + 100));

  // Vane pixels: brightly saturated yellow-green, blue, or red outside the paper
  const mask = new Uint8Array(width * height);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = (y * width + x) * 4;
      const [h, s, v] = rgbToHsv(rgba[i], rgba[i + 1], rgba[i + 2]);
      const isVane = (
        (h >= 45 && h <= 100 && s > 0.55 && v > 0.40) || // yellow-green (lime)
        (h >= 195 && h <= 245 && s > 0.50 && v > 0.30) || // blue
        ((h >= 345 || h <= 15) && s > 0.60 && v > 0.30)   // red
      );
      if (isVane) mask[y * width + x] = 1;
    }
  }

  // BFS connected-component labelling
  const labels = new Int32Array(width * height).fill(-1);
  const blobs: VaneBlob[] = [];
  const DIRS: [number, number][] = [[-1,0],[1,0],[0,-1],[0,1]];

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (!mask[y * width + x] || labels[y * width + x] >= 0) continue;
      const queue: number[] = [y * width + x];
      labels[y * width + x] = blobs.length;
      let head = 0, sumX = 0, sumY = 0, area = 0;
      while (head < queue.length) {
        const idx = queue[head++];
        const qy = Math.floor(idx / width), qx = idx % width;
        sumX += qx; sumY += qy; area++;
        if (area > MAX_AREA * 4) break; // abort oversized regions early
        for (const [dx, dy] of DIRS) {
          const nx = qx + dx, ny = qy + dy;
          if (nx < x0 || nx > x1 || ny < y0 || ny > y1) continue;
          const ni = ny * width + nx;
          if (!mask[ni] || labels[ni] >= 0) continue;
          labels[ni] = blobs.length;
          queue.push(ni);
        }
      }
      if (area >= MIN_AREA && area <= MAX_AREA)
        blobs.push({ cx: sumX / area, cy: sumY / area, area });
    }
  }
  return blobs;
}

// Match vane blobs to shaft nock endpoints
function matchVanes(
  arrows: { tip: Pt; nock: Pt | null }[],
  vanes: VaneBlob[],
  matchRadius: number,
): { tip: Pt; nock: Pt | null }[] {
  const usedVane = new Uint8Array(vanes.length);
  return arrows.map(arrow => {
    if (!arrow.nock) return arrow;
    const [nx, ny] = arrow.nock;
    const [tx, ty] = arrow.tip;
    const sdx = nx - tx, sdy = ny - ty, slen = Math.hypot(sdx, sdy);

    let bestDist = matchRadius, bestVane = -1;
    for (let vi = 0; vi < vanes.length; vi++) {
      if (usedVane[vi]) continue;
      const { cx, cy } = vanes[vi];
      // Vane must be roughly in the nock direction (past the shaft midpoint)
      if (slen > 0) {
        const dot = ((cx - tx) * sdx + (cy - ty) * sdy) / (slen * slen);
        if (dot < 0.4) continue;
      }
      const dist = Math.hypot(cx - nx, cy - ny);
      if (dist < bestDist) { bestDist = dist; bestVane = vi; }
    }
    if (bestVane >= 0) {
      usedVane[bestVane] = 1;
      return { tip: arrow.tip, nock: [vanes[bestVane].cx, vanes[bestVane].cy] };
    }
    return arrow;
  });
}

// ---------------------------------------------------------------------------
// P9-T5: Deduplicate — cluster tips within 15 px, keep longest shaft
// ---------------------------------------------------------------------------

function deduplicateTips(
  arrows: { tip: Pt; nock: Pt | null }[], clusterRadius: number,
): { tip: Pt; nock: Pt | null }[] {
  const byLen = arrows
    .map((a, i) => ({ a, i, len: a.nock ? Math.hypot(a.nock[0] - a.tip[0], a.nock[1] - a.tip[1]) : 0 }))
    .sort((x, y) => y.len - x.len);

  const used = new Uint8Array(arrows.length);
  const out: { tip: Pt; nock: Pt | null }[] = [];
  for (const { a, i } of byLen) {
    if (used[i]) continue;
    used[i] = 1;
    for (let j = 0; j < arrows.length; j++) {
      if (used[j]) continue;
      if (Math.hypot(arrows[j].tip[0] - a.tip[0], arrows[j].tip[1] - a.tip[1]) <= clusterRadius)
        used[j] = 1;
    }
    out.push(a);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main export: findArrows
// ---------------------------------------------------------------------------

export function findArrows(
  rgba: Uint8Array,
  width: number,
  height: number,
  result: ArcheryResult,
): ArrowDetection[] {
  if (!result.paperBoundary || result.rings.length === 0) return [];

  const boundary = result.paperBoundary.points;
  const rings    = result.rings;

  // Target centre (innermost ring centroid)
  const inner = rings[0];
  const cx = inner.points.reduce((s, p) => s + p[0], 0) / inner.points.length;
  const cy = inner.points.reduce((s, p) => s + p[1], 0) / inner.points.length;

  // Precompute ring radii for anti-ring filter
  const ringRadii = rings.map(ring => {
    const rcx = ring.points.reduce((s, p) => s + p[0], 0) / ring.points.length;
    const rcy = ring.points.reduce((s, p) => s + p[1], 0) / ring.points.length;
    return ring.points.reduce((s, p) => s + Math.hypot(p[0] - rcx, p[1] - rcy), 0) / ring.points.length;
  });

  // --- Hough in 2× downsampled space ---
  const SCALE = 2;
  const gray = buildGray(rgba, width, height);
  const { g: gd, w: wd, h: hd } = downsampleGray(gray, width, height, SCALE);
  const edges = sobelEdge(gd, wd, hd, 35);

  const bb = polyBBox(boundary);
  const margin = 80 / SCALE;

  // P9-T1: raw segments (downsampled coords)
  const rawDown = houghSegments(
    edges, wd, hd,
    bb.xMin / SCALE - margin, bb.xMax / SCALE + margin,
    bb.yMin / SCALE - margin, bb.yMax / SCALE + margin,
    /* minVotes */ 18, /* minLen */ 15, /* maxGap */ 5, /* maxPeaks */ 300,
  );

  // Scale up to full resolution
  let segs: Seg[] = rawDown.map(([[x0, y0], [x1, y1]]) =>
    [[x0 * SCALE, y0 * SCALE], [x1 * SCALE, y1 * SCALE]] as Seg,
  );

  // P9-T2a: collapse parallel edge-pairs into centerlines
  segs = mergeCenterlines(segs, /* angleTolDeg */ 5, /* perpTolPx */ 12);
  // P9-T2b: reassemble collinear halves split at crossings
  segs = mergeCollinear(segs, /* angleTolDeg */ 3, /* perpTolPx */ 6, /* gapTolPx */ 30);

  // P9-T3: length + area + anti-ring filter
  segs = filterSegments(segs, boundary, ringRadii, cx, cy, /* minLen */ 30, /* nearBoundaryTol */ 80);

  // Assign tip (closer to centre) / nock
  let arrows = assignTipNock(segs, cx, cy);

  // P9-T4: detect vanes and match to nock endpoints
  const vanes = detectVanes(rgba, width, height, boundary);
  arrows = matchVanes(arrows, vanes, /* matchRadius */ 90);

  // P9-T5: deduplicate by tip cluster
  arrows = deduplicateTips(arrows, /* clusterRadius */ 15);

  // P9-T6: hole fallback — reserved for next iteration

  return arrows.map(a => ({ tip: a.tip as [number, number], nock: a.nock as [number, number] | null }));
}
