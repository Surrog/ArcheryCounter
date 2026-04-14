import { findTarget, pointInPolygon } from './targetDetection';
import { decodeBase64Jpeg } from './imageLoader';
import { detectArrowsNN } from './arrowDetector';
import type { EllipseData, TargetBoundary, ColourCalibration, Pixel } from './targetDetection';
import type { SplineRing } from './spline';
import type { ScoredArrow } from './scoring';

export type { SplineRing };
export type { EllipseData, Pixel, TargetBoundary, ColourCalibration };
export type { ScoredArrow };

export interface TargetResult {
  rings: SplineRing[];
  paperBoundary: [number, number][];
}

export interface ProcessImageResult {
  /** Per-target results (rings + boundary). Length ≥ 1 on success. */
  targets: TargetResult[];
  /** All detected arrows, scored against the nearest target. Tips outside all boundaries are discarded. */
  arrows: ScoredArrow[];
  // --- backwards-compat single-target fields (targets[0]) ---
  rings: SplineRing[];
  paperBoundary?: TargetBoundary;
  calibration?: ColourCalibration;
  /** Raw per-ray transition points for each ring (index 0 = innermost). */
  ringPoints?: Pixel[][];
}

const ArcheryCounter = {
  async processImage(
    imageUri: string,
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
      rings: result.rings,
      paperBoundary: result.paperBoundary,
      calibration: result.calibration,
      ringPoints: result.ringPoints,
    };
  },
};

export default ArcheryCounter;
