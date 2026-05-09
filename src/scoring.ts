/**
 * Phase 10 — Arrow scoring pipeline.
 * See docs/research.md §5 for design rationale.
 *
 * scoreArrow        — geometric score from ring splines (primary)
 * samplePatchZone   — colour zone at tip (cross-check)
 * scoreArrowWithCheck — combines both; sets lowConfidence on disagreement
 */

import { pointInClosedSpline, splineCentroid, splineRadius } from './spline';
import type { SplineRing } from './spline';
import { classifyPixelZone, rgbToHsv } from './targetDetection';
import type { ColourCalibration, ZoneName } from './targetDetection';

export interface ScoredArrow {
  tip:            [number, number];
  score:          number | 'X';   // 0 = miss, 1–10 or 'X'
  lowConfidence?: boolean;        // geometric and colour zone disagree
}

function isScore(x: unknown): x is number | 'X' {
  return (typeof x === "number" &&
    Number.isInteger(x) &&
    x >= 0 && x <= 10) ||
    x === 'X';
}

export function isScoredArrow(x: unknown): x is ScoredArrow {
  return typeof x === "object" && x != null &&
    Array.isArray((x as ScoredArrow).tip) &&
    (x as ScoredArrow).tip.length === 2 &&
    typeof (x as ScoredArrow).tip[0] === "number" &&
    typeof (x as ScoredArrow).tip[1] === "number" &&
    isScore((x as ScoredArrow).score) &&
    ((x as ScoredArrow).lowConfidence === undefined ||
      typeof (x as ScoredArrow).lowConfidence === "boolean");
}

// Zone → [minScore, maxScore] for consistency check
const ZONE_SCORE_RANGE: Record<ZoneName, [number, number]> = {
  gold:  [9, 10],
  red:   [7, 8],
  blue:  [5, 6],
  black: [3, 4],
  white: [1, 2],
};

/**
 * Geometric score for a tip point against the 10 detected ring boundaries.
 *
 * Walks rings[0..9] from innermost outward; returns 10 - i for the first ring
 * that contains the tip, or 0 (miss) if outside all rings.
 *
 * X-ring: inside ring[0] AND distance from target centre < 0.4 × ring[1] radius.
 */
export function scoreArrow(tip: [number, number], rings: SplineRing[]): number | 'X' {
  for (let i = 0; i < rings.length; i++) {
    if (pointInClosedSpline(tip, rings[i].points)) {
      if (i === 0 && rings.length >= 2) {
        const centre    = splineCentroid(rings[0]);
        const goldOuter = splineRadius(rings[1]);
        if (Math.hypot(tip[0] - centre[0], tip[1] - centre[1]) < 0.4 * goldOuter) {
          return 'X';
        }
      }
      return 10 - i;
    }
  }
  return 0;
}

/**
 * Samples an annular patch (inner r=4, outer r=12) around the tip and returns
 * the modal colour zone, or null if no classifiable pixels are found.
 *
 * Excludes hay/straw pixels (S < 0.15).
 */
export function samplePatchZone(
  rgba: Uint8Array, width: number, height: number,
  tip: [number, number],
  cal: ColourCalibration,
): ZoneName | null {
  const [tx, ty] = tip;
  const INNER_R2 = 4 * 4;
  const OUTER_R2 = 12 * 12;
  const counts = new Map<ZoneName, number>();

  for (let dy = -12; dy <= 12; dy++) {
    for (let dx = -12; dx <= 12; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 < INNER_R2 || d2 > OUTER_R2) continue;
      const px = Math.round(tx + dx), py = Math.round(ty + dy);
      if (px < 0 || px >= width || py < 0 || py >= height) continue;
      const idx = (py * width + px) * 4;
      const [h, s, v] = rgbToHsv(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
      if (s < 0.15) continue; // exclude hay/straw
      const zone = classifyPixelZone([h, s, v], cal);
      if (zone) counts.set(zone, (counts.get(zone) ?? 0) + 1);
    }
  }

  let best: ZoneName | null = null, bestCount = 0;
  for (const [z, c] of counts) {
    if (c > bestCount) { bestCount = c; best = z; }
  }
  return best;
}

