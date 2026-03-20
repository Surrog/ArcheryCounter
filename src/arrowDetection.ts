/**
 * Phase 9 — Arrow detection pipeline.
 * See docs/plan.md §Phase 9 for design rationale.
 *
 * Pipeline stages:
 *   [D1]  Hough segment extraction — raw segment count from shaftMask
 *   [D2]  Collinear merge — segment count after each mergeCollinear pass
 *   [D3]  filterSegments — per-rejection-reason counts + per-survivor PASS lines
 *   [D4]  After verifyDarkStripe + second merge + length filter
 *   [D5]  deduplicateTips — which tips were suppressed and why
 *   [D6]  removeMidshaftDuplicates — which arrows were suppressed (case 1/2/3)
 *   [D7]  matchVanes — vane blob count + per-arrow vane match results
 *   [D8]  Final arrow list
 *
 * Enable debug output:
 *   DEBUG_ARROWS=1 npx jest groundTruth 2>&1 | grep '\[D6\]'
 *
 * All stages write to stderr so Jest's own stdout output is unaffected.
 * Stage labels are prefixed [D1]–[D8]; grep isolates individual stages:
 *   DEBUG_ARROWS=1 npx jest 2>&1 | grep -E '\[D[1-8]\]'
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
// Step 1–2: Sobel edge detection for Hough input
// ---------------------------------------------------------------------------

/**
 * Zone-adaptive relative dark mask for shaft detection.
 *
 * Arrow shafts (carbon fibre, V≈0.1–0.3) are significantly darker than any
 * scoring zone background: gold (V≈0.85), red (V≈0.70), blue (V≈0.55),
 * white (V≈0.90).  Marking pixels that fall below 65 % of their zone's
 * expected V catches shafts while rejecting zone-colour background pixels,
 * ring-arc edge pixels (which are the zone's own colour), and hay-bale
 * pixels (excluded because they are outside the paper boundary).
 *
 * The black zone (V≈0.15) is EXCLUDED: shaft V≈0.15 is
 * indistinguishable from the black background there.  The collinear merge
 * bridges the resulting gap in the middle of the shaft.
 */
function buildRelativeDarkMask(
  rgba: Uint8Array, width: number, height: number,
  boundary: Pt[], ringRadii: number[], cx: number, cy: number,
  calibration: { gold: [number,number,number]; red: [number,number,number];
                 blue: [number,number,number]; white: [number,number,number] } | undefined,
): Uint8Array {
  const RATIO = 0.75; // shaft must be darker than RATIO * zone_V
  const goldThr  = (calibration ? calibration.gold[2]  : 0.85) * RATIO;
  const redThr   = (calibration ? calibration.red[2]   : 0.70) * RATIO;
  const blueThr  = (calibration ? calibration.blue[2]  : 0.55) * RATIO;
  const whiteThr = (calibration ? calibration.white[2] : 0.90) * RATIO;
  const outerThr = 0.30; // outside scoring rings but still within paper boundary

  // Ring-zone boundaries
  const rGold  = ringRadii.length > 1 ? ringRadii[1] : 36;
  const rRed   = ringRadii.length > 3 ? ringRadii[3] : 73;
  const rBlue  = ringRadii.length > 5 ? ringRadii[5] : 110;
  const rBlack = ringRadii.length > 7 ? ringRadii[7] : 145;
  const rWhite = ringRadii.length > 9 ? ringRadii[9] : 182;

  const mask = new Uint8Array(width * height);
  const bb = polyBBox(boundary);
  const x0 = Math.max(0, Math.floor(bb.xMin));
  const x1 = Math.min(width  - 1, Math.ceil(bb.xMax));
  const y0 = Math.max(0, Math.floor(bb.yMin));
  const y1 = Math.min(height - 1, Math.ceil(bb.yMax));

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (!ptInPoly(x, y, boundary)) continue;
      const r = Math.hypot(x - cx, y - cy);
      // Determine threshold for this zone; skip black zone
      let thr: number;
      if      (r <= rGold)  thr = goldThr;
      else if (r <= rRed)   thr = redThr;
      else if (r <= rBlue)  thr = blueThr;
      else if (r <= rBlack) continue;  // black zone: shaft V≈background V, indistinguishable → skip
      else if (r <= rWhite) thr = whiteThr;
      else                  thr = outerThr;

      const i4 = (y * width + x) * 4;
      const [, , v] = rgbToHsv(rgba[i4], rgba[i4 + 1], rgba[i4 + 2]);
      if (v < thr) mask[y * width + x] = 255;
    }
  }
  return mask;
}

/** Mark pixels darker than vThreshold (V channel in HSV) as 255, rest 0.  */
function buildDarkMask(rgba: Uint8Array, w: number, h: number, vThreshold: number): Uint8Array {
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const [, , v] = rgbToHsv(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    if (v < vThreshold) mask[i] = 255;
  }
  return mask;
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

  // Find local maxima with non-max suppression (±3 theta, ±5 r).
  // A tighter r window (was ±8) prevents strong bullseye-centre peaks from
  // suppressing nearby shaft peaks (arrows 2 and 6 whose ρ is only 7 px from
  // the dense-dark bullseye column).
  const peaks: { t: number; r: number; v: number }[] = [];
  for (let t = 0; t < N_THETA; t++) {
    for (let r = 0; r < nR; r++) {
      const v = acc[t * nR + r];
      if (v < minVotes) continue;
      let isMax = true;
      outer: for (let dt = -3; dt <= 3 && isMax; dt++) {
        for (let dr = -3; dr <= 3 && isMax; dr++) {
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
  minLen: number,
  debug = false,
): Seg[] {
  // Rejection counters — visible in [D3] debug output.
  // Use these to understand which filter is removing the most segments:
  //   len       — segment too short (< minLen)
  //   tipOOB    — tip endpoint is outside the paper boundary (hay-bale / frame lines)
  //   nockFar   — nock is > 120 px outside the paper (hay-bale lines exit far)
  //   radial    — radial profile shows a U-shape interior minimum (ring arc tangent)
  //   antiRing  — tip + nock both sit at a ring radius at a tangent angle (ring arc)
  let nLen = 0, nTipOOB = 0, nNockFar = 0, nRadial = 0, nAntiRing = 0;

  const out: Seg[] = [];
  for (const seg of segs) {
    const r = (reason: string) => { if (debug) console.error(`  [D3] REJECT ${reason} tip=(${Math.round(tipX0)},${Math.round(tipY0)}) len=${Math.round(segLen(seg))}`); };

    if (segLen(seg) < minLen) { nLen++; continue; }

    // Assign tip (closer to centre) and nock (farther from centre)
    const d0c = Math.hypot(seg[0][0] - cx, seg[0][1] - cy);
    const d1c = Math.hypot(seg[1][0] - cx, seg[1][1] - cy);
    const [tipX0, tipY0, nockX0, nockY0] = d0c <= d1c
      ? [seg[0][0], seg[0][1], seg[1][0], seg[1][1]]
      : [seg[1][0], seg[1][1], seg[0][0], seg[0][1]];

    // Tip must be inside the paper boundary (where arrows land).
    // Hay-bale lines approaching from outside have their closer endpoint
    // outside or just on the boundary.
    if (!ptInPoly(tipX0, tipY0, boundary)) { nTipOOB++; r('tip-OOB'); continue; }

    // Nock: if outside the boundary, it must be within 120 px of the boundary.
    // Real arrows have nocks just outside the paper (< 100 px outside);
    // hay-bale lines extend 150-400 px beyond the boundary.
    if (!ptInPoly(nockX0, nockY0, boundary)) {
      if (distToBoundary(nockX0, nockY0, boundary) > 120) { nNockFar++; r('nock-far'); continue; }
    }

    const ep0In = ptInPoly(seg[0][0], seg[0][1], boundary);
    const ep1In = ptInPoly(seg[1][0], seg[1][1], boundary);

    // Radial profile check for segments entirely inside the paper.
    // Sample 7 points and compute per-sample distance from target centre.
    if (ep0In && ep1In) {
      const NUM_R = 7;
      const rads: number[] = [];
      for (let k = 0; k < NUM_R; k++) {
        const t = k / (NUM_R - 1);
        const px = seg[0][0] + t * (seg[1][0] - seg[0][0]);
        const py = seg[0][1] + t * (seg[1][1] - seg[0][1]);
        rads.push(Math.hypot(px - cx, py - cy));
      }
      const minD = Math.min(...rads);

      // Minimum distance in the INTERIOR of the segment → ring arc tangent.
      // For Hough tangent lines, the tangent point (= closest point to centre)
      // falls in the MIDDLE of the detected segment; for arrow shafts the tip
      // is at one endpoint so the minimum is at k=0 or k=6.
      const tipRad = d0c <= d1c ? rads[0] : rads[NUM_R - 1];
      const minIdx = rads.indexOf(minD);
      if (minIdx >= 2 && minIdx <= NUM_R - 3 && tipRad < 2 * minD) {
        nRadial++;
        if (debug) console.error(`  [D3] REJECT radial-interior tip=(${Math.round(tipX0)},${Math.round(tipY0)}) minIdx=${minIdx} tipR=${Math.round(tipRad)} minR=${Math.round(minD)}`);
        continue;
      }
    }

    // Anti-ring: check the TIP endpoint (closer to centre) rather than the midpoint.
    // Only reject if the NOCK is also near the same ring radius — real arrow shafts have
    // their nock far from the tip's ring, whereas ring arcs have both endpoints at the ring.
    const tipDist = Math.min(d0c, d1c);
    const nockDist = Math.max(d0c, d1c);
    const segAng = segAngle(seg);
    const tangentAngTip = Math.atan2(tipY0 - cy, tipX0 - cx) + Math.PI / 2;
    const tNormTip = ((tangentAngTip % Math.PI) + Math.PI) % Math.PI;
    let rejected = false;
    for (const rr of ringRadii) {
      if (Math.abs(tipDist - rr) < 25 && angleDiff(segAng, tNormTip) < 35 * Math.PI / 180) {
        if (Math.abs(nockDist - rr) > 15) continue; // nock is far from this ring → real arrow
        nAntiRing++;
        if (debug) console.error(`  [D3] REJECT anti-ring-tip tip=(${Math.round(tipX0)},${Math.round(tipY0)}) rr=${Math.round(rr)} tipR=${Math.round(tipDist)} nockR=${Math.round(nockDist)}`);
        rejected = true; break;
      }
    }
    if (rejected) continue;

    // Belt-and-suspenders: also check midpoint (catches arcs whose endpoints are not at the ring)
    const mx = (seg[0][0] + seg[1][0]) / 2;
    const my = (seg[0][1] + seg[1][1]) / 2;
    const dToCenter = Math.hypot(mx - cx, my - cy);
    for (const rr of ringRadii) {
      if (Math.abs(dToCenter - rr) < 25) {
        if (Math.abs(nockDist - rr) > 20) continue; // nock far from ring → real arrow
        const tangentAng = Math.atan2(my - cy, mx - cx) + Math.PI / 2;
        const tNorm = ((tangentAng % Math.PI) + Math.PI) % Math.PI;
        if (angleDiff(segAng, tNorm) < 35 * Math.PI / 180) {
          nAntiRing++;
          if (debug) console.error(`  [D3] REJECT anti-ring-mid tip=(${Math.round(tipX0)},${Math.round(tipY0)}) rr=${Math.round(rr)} midR=${Math.round(dToCenter)}`);
          rejected = true; break;
        }
      }
    }
    if (!rejected) out.push(seg);
  }

  if (debug) {
    console.error(`[D3] filterSegments: ${segs.length} → ${out.length}` +
      ` (len=${nLen} tipOOB=${nTipOOB} nockFar=${nNockFar} radial=${nRadial} antiRing=${nAntiRing})`);
    for (const seg of out) {
      const d0c = Math.hypot(seg[0][0]-cx, seg[0][1]-cy);
      const d1c = Math.hypot(seg[1][0]-cx, seg[1][1]-cy);
      const [tx,ty] = d0c<=d1c ? [seg[0][0],seg[0][1]] : [seg[1][0],seg[1][1]];
      const [nx,ny] = d0c<=d1c ? [seg[1][0],seg[1][1]] : [seg[0][0],seg[0][1]];
      console.error(`  PASS tip=(${Math.round(tx)},${Math.round(ty)}) nock=(${Math.round(nx)},${Math.round(ny)}) len=${Math.round(segLen(seg))} tipR=${Math.round(Math.min(d0c,d1c))}`);
    }
  }
  return out;
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
  targetCx: number, targetCy: number,
  debug = false,
): { tip: Pt; nock: Pt | null }[] {
  if (debug) console.error(`[D7] matchVanes: ${vanes.length} vane blobs, ${arrows.length} arrows`);
  const usedVane = new Uint8Array(vanes.length);
  return arrows.map(arrow => {
    if (!arrow.nock) return arrow;
    const [nx, ny] = arrow.nock;
    const [tx, ty] = arrow.tip;
    const sdx = nx - tx, sdy = ny - ty, slen = Math.hypot(sdx, sdy);
    const tipR = Math.hypot(tx - targetCx, ty - targetCy);

    let bestDist = matchRadius, bestVane = -1;
    for (let vi = 0; vi < vanes.length; vi++) {
      if (usedVane[vi]) continue;
      const { cx, cy } = vanes[vi];
      // Vane must be farther from target centre than the tip: real arrow vanes
      // are at the hay-bale end (high r), while scoring-zone colours (gold) can
      // create false vane blobs deep inside the target (low r).
      if (Math.hypot(cx - targetCx, cy - targetCy) < tipR) continue;
      // Vane must be close to the shaft line (not a target-face colour patch)
      if (slen > 0 && perpDist(cx, cy, tx, ty, nx, ny) > 20) continue;
      // Vane must be roughly in the nock direction (past the shaft midpoint)
      if (slen > 0) {
        const dot = ((cx - tx) * sdx + (cy - ty) * sdy) / (slen * slen);
        if (dot < 0.4) continue;
      }
      const dist = Math.hypot(cx - nx, cy - ny);
      if (dist < bestDist) { bestDist = dist; bestVane = vi; }
    }
    if (bestVane >= 0) {
      if (debug) console.error(`  [D7] match tip=(${Math.round(tx)},${Math.round(ty)}) nock=(${Math.round(nx)},${Math.round(ny)}) → vane=(${Math.round(vanes[bestVane].cx)},${Math.round(vanes[bestVane].cy)}) dist=${bestDist.toFixed(1)}`);
      usedVane[bestVane] = 1;
      return { tip: arrow.tip, nock: [vanes[bestVane].cx, vanes[bestVane].cy] };
    }
    return arrow;
  });
}

// ---------------------------------------------------------------------------
// Shaft pixel verification: arrow shafts (carbon fibre) appear as dark stripes
// on the brighter scoring-zone background.  Sample along the candidate segment
// centre and check that centre pixels are consistently darker than the
// perpendicular background.  Ring-arc detections (colour transitions between
// bright scoring zones) fail this check; even the black/white zone boundary
// does not produce a dark stripe flanked by lighter pixels on BOTH sides.
// ---------------------------------------------------------------------------

// Verify a shaft segment by checking that there are dark pixels (V < 0.45)
// along the segment centerline.  Carbon-fibre arrow shafts are always dark
// regardless of which scoring zone they pass through.  Ring-arc detections sit
// at a colour-zone boundary: gold/red, red/blue, and black/white transitions
// are NOT dark at their midline; only the blue/black boundary is dark, but
// that is caught by the anti-ring filter.
//
// A ±4 px perpendicular sweep compensates for the ≈2–4 px coordinate error
// introduced by 2× downsampling + centerline averaging.
function verifyDarkStripe(
  seg: Seg, rgba: Uint8Array, width: number, height: number,
): boolean {
  const dx = seg[1][0] - seg[0][0], dy = seg[1][1] - seg[0][1];
  const len = Math.hypot(dx, dy);
  if (len < 1) return true;
  const nx = -dy / len, ny = dx / len; // unit perpendicular

  const N = 7; // 6 interior sample points
  let darkCount = 0, validCount = 0;

  for (let k = 1; k < N; k++) {
    const t = k / N;
    const px = seg[0][0] + t * dx;
    const py = seg[0][1] + t * dy;

    // Find minimum V in ±6 px perpendicular sweep.
    // Wider sweep compensates for DS→full-res quantisation (shaft may be 1 px
    // wide in DS, so the centerline merge position can be 3–5 px off the dark core).
    let minV = Infinity;
    for (let off = -6; off <= 6; off++) {
      const sx = Math.round(px + off * nx);
      const sy = Math.round(py + off * ny);
      if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
      const i = (sy * width + sx) * 4;
      const [, , v] = rgbToHsv(rgba[i], rgba[i + 1], rgba[i + 2]);
      if (v < minV) minV = v;
    }

    if (isFinite(minV)) {
      validCount++;
      if (minV < 0.55) darkCount++;
    }
  }

  // Require ≥ 40% of sample points to have a dark pixel within ±6 px.
  // Gold/red ring arc midlines (V ≈ 0.8) and red/blue midlines (V ≈ 0.6)
  // have no dark pixels in their vicinity → rejected.
  // Blue/black and black/white boundaries may survive here but are caught by
  // the anti-ring filter.
  return validCount === 0 || darkCount >= Math.ceil(validCount * 0.4);
}

// ---------------------------------------------------------------------------
// Remove midshaft duplicates — suppress shorter arrows whose tip lies on
// a longer arrow's shaft line (perpendicular distance < tol).  Handles the
// case where a Hough peak picks up only the lower half of a shaft, producing
// a second "arrow" whose tip is actually a midshaft point of the real arrow.
// ---------------------------------------------------------------------------

function removeMidshaftDuplicates(
  arrows: { tip: Pt; nock: Pt | null }[],
  perpTolPx: number, angleTolDeg: number,
  debug = false,
): { tip: Pt; nock: Pt | null }[] {
  const angleTol = angleTolDeg * Math.PI / 180;
  const byLen = [...arrows]
    .map((a, i) => ({ a, i, len: a.nock ? Math.hypot(a.nock[0] - a.tip[0], a.nock[1] - a.tip[1]) : 0 }))
    .sort((x, y) => y.len - x.len);
  const suppressed = new Uint8Array(arrows.length);
  for (const { a, i } of byLen) {
    if (suppressed[i] || !a.nock) continue;
    const angA = Math.atan2(a.nock[1] - a.tip[1], a.nock[0] - a.tip[0]);
    for (const { a: b, i: j } of byLen) {
      if (j === i || suppressed[j]) continue;
      const bNock = b.nock ?? b.tip;
      const angB = Math.atan2(bNock[1] - b.tip[1], bNock[0] - b.tip[0]);
      let da = Math.abs(angA - angB) % (2 * Math.PI);
      if (da > Math.PI) da = 2 * Math.PI - da;
      if (da > Math.PI / 2) da = Math.PI - da;
      const pd = perpDist(b.tip[0], b.tip[1], a.tip[0], a.tip[1], a.nock[0], a.nock[1]);
      // Case 1: collinear fragment — same direction, tip on shaft (t ∈ [-0.05, 1.05])
      if (da <= angleTol && pd < perpTolPx) {
        const sdx = a.nock[0] - a.tip[0], sdy = a.nock[1] - a.tip[1];
        const sLen2 = sdx * sdx + sdy * sdy;
        const t = sLen2 < 1 ? 0 : ((b.tip[0] - a.tip[0]) * sdx + (b.tip[1] - a.tip[1]) * sdy) / sLen2;
        if (t >= -0.05 && t <= 1.05) {
          if (debug) console.error(`  [D6] Case1 suppress tip=(${Math.round(b.tip[0])},${Math.round(b.tip[1])}) da=${(da*180/Math.PI).toFixed(1)}° pd=${pd.toFixed(1)} t=${t.toFixed(2)} by=(${Math.round(a.tip[0])},${Math.round(a.tip[1])})`);
          suppressed[j] = 1; continue;
        }
      }
      // Case 2: intersection artifact — tip crosses another arrow's shaft at a
      // large angle (≥ 30°).  Nearly-collinear pairs (da < 30°) are handled by
      // case 1 only; this avoids suppressing shafts that run nearly parallel.
      if (da >= 30 * Math.PI / 180 && pd < 10) {
        const sdx = a.nock[0] - a.tip[0], sdy = a.nock[1] - a.tip[1];
        const sLen2 = sdx * sdx + sdy * sdy;
        const t = sLen2 < 1 ? 0 : ((b.tip[0] - a.tip[0]) * sdx + (b.tip[1] - a.tip[1]) * sdy) / sLen2;
        if (t >= 0 && t <= 1) {
          if (debug) console.error(`  [D6] Case2 suppress tip=(${Math.round(b.tip[0])},${Math.round(b.tip[1])}) da=${(da*180/Math.PI).toFixed(1)}° pd=${pd.toFixed(1)} t=${t.toFixed(2)} by=(${Math.round(a.tip[0])},${Math.round(a.tip[1])})`);
          suppressed[j] = 1;
        }
      }
      // Case 3: nock-sharing fragment — two segments from the same physical
      // arrow share nearly the same nock point.  The longer detection (a) is
      // kept; the shorter fragment (b) is suppressed when its tip lies along
      // the shaft of a.
      if (a.nock && b.nock) {
        const nockDist = Math.hypot(a.nock[0] - b.nock[0], a.nock[1] - b.nock[1]);
        if (nockDist < 20) {
          const sdx = a.nock[0] - a.tip[0], sdy = a.nock[1] - a.tip[1];
          const sLen2 = sdx * sdx + sdy * sdy;
          const t = sLen2 < 1 ? 0 : ((b.tip[0] - a.tip[0]) * sdx + (b.tip[1] - a.tip[1]) * sdy) / sLen2;
          if (t >= 0 && t <= 1) {
            if (debug) console.error(`  [D6] Case3 suppress tip=(${Math.round(b.tip[0])},${Math.round(b.tip[1])}) nockDist=${nockDist.toFixed(1)} t=${t.toFixed(2)} by=(${Math.round(a.tip[0])},${Math.round(a.tip[1])})`);
            suppressed[j] = 1; continue;
          }
        }
      }
    }
  }
  if (debug) console.error(`[D6] removeMidshaftDuplicates: ${arrows.length} → ${arrows.filter((_, i) => !suppressed[i]).length}`);
  return arrows.filter((_, i) => !suppressed[i]);
}

// ---------------------------------------------------------------------------
// P9-T5: Deduplicate — cluster tips within 15 px, keep longest shaft
// ---------------------------------------------------------------------------

function deduplicateTips(
  arrows: { tip: Pt; nock: Pt | null }[], clusterRadius: number,
  debug = false,
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
      const d = Math.hypot(arrows[j].tip[0] - a.tip[0], arrows[j].tip[1] - a.tip[1]);
      if (d <= clusterRadius) {
        if (debug) console.error(`  [D5] suppress tip=(${Math.round(arrows[j].tip[0])},${Math.round(arrows[j].tip[1])}) dist=${d.toFixed(1)}px from keeper=(${Math.round(a.tip[0])},${Math.round(a.tip[1])})`);
        used[j] = 1;
      }
    }
    out.push(a);
  }
  if (debug) console.error(`[D5] deduplicateTips: ${arrows.length} → ${out.length}`);
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

  // --- Zone-adaptive relative-dark-pixel Hough ---
  // Shaft pixels (V≈0.1–0.3) are marked only where they are significantly
  // darker than the expected zone background.  This excludes hay-bale pixels
  // (outside paper boundary), ring-arc edge pixels (zone colour, not dark),
  // and the black zone (where shaft/background are indistinguishable).
  // Each shaft contributes ~shaft_length × shaft_width × density votes to one
  // Hough bin → 50–200+ votes, safely above minVotes=30.
  // Ring arcs: even in white/blue zones the entire arc arc contributes only
  // ~16 votes per tangent direction → below minVotes → not detected.
  const shaftMask = buildRelativeDarkMask(
    rgba, width, height, boundary, ringRadii, cx, cy,
    result.calibration,
  );
  const darkMask = buildDarkMask(rgba, width, height, /* vThreshold */ 0.45);
  const wd = width, hd = height;

  const bb = polyBBox(boundary);
  const margin = 80;

  // P9-T1: raw segments
  const rawDown = houghSegments(
    shaftMask, wd, hd,
    bb.xMin - margin, bb.xMax + margin,
    bb.yMin - margin, bb.yMax + margin,
    /* minVotes */ 15, /* minLen */ 30, /* maxGap */ 8, /* maxPeaks */ 500,
  );
  const DEBUG = !!process.env.DEBUG_ARROWS;
  if (DEBUG) console.error(`[D1] Hough raw: ${rawDown.length} segs  cx=(${Math.round(cx)},${Math.round(cy)}) ringRadii=${ringRadii.map(r=>Math.round(r)).join(',')}`);

  let segs: Seg[] = rawDown;

  // P9-T2a: centerline merge skipped — dark pixels are at the shaft centre,
  // not at edges, so there's no edge-pair to merge.
  // P9-T2b: reassemble shaft halves split at crossings or across zone gaps
  // (gap ≤ 60 px: upper blue-zone end to lower white-zone start can be ~57 px)
  segs = mergeCollinear(segs, /* angleTolDeg */ 3, /* perpTolPx */ 6, /* gapTolPx */ 60);
  if (DEBUG) console.error(`[D2] after first mergeCollinear: ${segs.length} segs`);

  segs = filterSegments(segs, boundary, ringRadii, cx, cy, /* minLen */ 30, DEBUG);
  // Reject segments with no continuous dark stripe (V < 0.55 within ±6 px of centerline).
  // This eliminates false positives whose Hough votes come from scattered dark pixels
  // (e.g. scattered JPEG artefacts in the gold zone at V=0.55–0.64) rather than a real shaft.
  segs = segs.filter(seg => verifyDarkStripe(seg, rgba, width, height));
  // Second collinear merge: collapse shaft fragments that survived independent filtering.
  // Looser tolerances to handle Hough-bin quantisation differences between fragments.
  segs = mergeCollinear(segs, /* angleTolDeg */ 8, /* perpTolPx */ 3, /* gapTolPx */ 150);
  // Drop any short fragments that remain after the second merge.
  segs = segs.filter(seg => segLen(seg) >= 50);
  if (DEBUG) {
    console.error(`[D4] after darkStripe+merge2+len50: ${segs.length} segs`);
    for (const s of segs.slice().sort((a, b) => segLen(b) - segLen(a)).slice(0, 20)) {
      const d0c = Math.hypot(s[0][0]-cx, s[0][1]-cy), d1c = Math.hypot(s[1][0]-cx, s[1][1]-cy);
      const [tx, ty] = d0c <= d1c ? [s[0][0], s[0][1]] : [s[1][0], s[1][1]];
      const [nx, ny] = d0c <= d1c ? [s[1][0], s[1][1]] : [s[0][0], s[0][1]];
      console.error(`  [D4] tip=(${Math.round(tx)},${Math.round(ty)}) nock=(${Math.round(nx)},${Math.round(ny)}) len=${Math.round(segLen(s))} tipR=${Math.round(Math.min(d0c,d1c))}`);
    }
  }

  // Assign tip (closer to centre) / nock
  let arrows = assignTipNock(segs, cx, cy);

  // Filter arrows whose tip is outside the outermost scored ring.
  // Real arrow tips penetrate the target face; hay-bale lines and other
  // artifacts can have their closer endpoint well outside the ring system
  // even if still inside the paper boundary.
  if (ringRadii.length >= 10) {
    const outerR = ringRadii[9];
    arrows = arrows.filter(a => Math.hypot(a.tip[0] - cx, a.tip[1] - cy) <= outerR);
  }

  // P9-T5: deduplicate and remove midshaft duplicates BEFORE vane matching
  // so that shaft-direction comparison uses the raw Hough segment geometry.
  arrows = deduplicateTips(arrows, /* clusterRadius */ 15, DEBUG);
  arrows = removeMidshaftDuplicates(arrows, /* perpTolPx */ 8, /* angleTolDeg */ 15, DEBUG);

  // P9-T4: detect vanes and match to nock endpoints
  const vanes = detectVanes(rgba, width, height, boundary);
  arrows = matchVanes(arrows, vanes, /* matchRadius */ 90, cx, cy, DEBUG);

  if (DEBUG) {
    console.error(`[D8] final: ${arrows.length} arrows`);
    for (const a of arrows) {
      const len = a.nock ? Math.hypot(a.nock[0]-a.tip[0], a.nock[1]-a.tip[1]) : 0;
      const tipR = Math.hypot(a.tip[0]-cx, a.tip[1]-cy);
      const nockStr = a.nock ? `(${Math.round(a.nock[0])},${Math.round(a.nock[1])})` : 'null';
      console.error(`  [D8] tip=(${Math.round(a.tip[0])},${Math.round(a.tip[1])}) nock=${nockStr} len=${Math.round(len)} tipR=${Math.round(tipR)}`);
    }
  }

  // P9-T6: hole fallback — reserved for next iteration

  return arrows.map(a => ({ tip: a.tip as [number, number], nock: a.nock as [number, number] | null }));
}
