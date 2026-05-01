import { findTarget, pointInPolygon } from './targetDetection';
import { decodeBase64Jpeg } from './imageLoader';
import { detectArrowsNN } from './arrowDetector';
import type { TargetBoundary, ColourCalibration, Pixel } from './targetDetection';
import { SplineRing, isSplineRing } from './spline';
import { ScoredArrow, isScoredArrow } from './scoring';

export type { SplineRing };
export type { Pixel, TargetBoundary, ColourCalibration };
export type { ScoredArrow };

export interface TargetResult {
  rings: SplineRing[];
  paperBoundary: [number, number][];
}

export function isTargetResult(x: unknown): x is TargetResult {
  return typeof x === "object" && x != null &&
    Array.isArray((x as TargetResult).rings) &&
    (x as TargetResult).rings.every(ring => isSplineRing(ring)) &&
    Array.isArray((x as TargetResult).paperBoundary) &&
    (x as TargetResult).paperBoundary.every(([px, py]) => typeof px === "number" && typeof py === "number");
}

export interface ProcessImageResult {
  /** Per-target results (rings + boundary). Length ≥ 1 on success. */
  targets: TargetResult[];
  /** All detected arrows, scored against the nearest target. Tips outside all boundaries are discarded. */
  arrows: ScoredArrow[];
}

const ArcheryCounter = {
  async processImage(
    base64: string,
    options?: { modelPath?: string },
  ): Promise<ProcessImageResult> {
    const { rgba, width, height } = decodeBase64Jpeg(base64);
    const result = findTarget(rgba, width, height);
    if (!result.success) throw new Error(result.error ?? 'Detection failed');

    const targets: TargetResult[] = result.targets.map(t => ({
      rings: t.rings,
      paperBoundary: t.paperBoundary.points as [number, number][],
    }));

    let arrows: ScoredArrow[] = [];
    if (options?.modelPath) {
      try {
        const rawArrows = await detectArrowsNN(rgba, width, height, options.modelPath);
        // Discard tips that fall outside every target boundary.
        arrows = rawArrows.filter(a =>
          targets.some(t =>
            pointInPolygon({ x: a.tip[0], y: a.tip[1] }, { points: t.paperBoundary }),
          ),
        );
      } catch (e) {
        // NN unavailable or failed — return empty arrow list
        console.warn('[ArcheryCounter] Arrow detection failed:', e);
      }
    }

    return {
      targets,
      arrows,
    };
  },
};

export default ArcheryCounter;
