import { findTarget } from './targetDetection';
import { decodeBase64Jpeg } from './imageLoader';
import { detectArrowsNN } from './arrowDetector';
import type { EllipseData, TargetBoundary, ColourCalibration, Pixel } from './targetDetection';
import type { SplineRing } from './spline';
import type { ScoredArrow } from './scoring';

export type { SplineRing as RingEllipse };
export type { SplineRing };
export type { EllipseData, Pixel, TargetBoundary, ColourCalibration };
export type { ScoredArrow };

export interface ProcessImageResult {
  rings: SplineRing[];
  paperBoundary?: TargetBoundary;
  calibration?: ColourCalibration;
  arrows: ScoredArrow[];
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

    let arrows: ScoredArrow[] = [];
    if (options?.modelPath) {
      try {
        arrows = await detectArrowsNN(rgba, width, height, options.modelPath);
      } catch {
        // NN unavailable or failed — return empty arrow list
      }
    }

    return { rings: result.rings, paperBoundary: result.paperBoundary, calibration: result.calibration, arrows, ringPoints: result.ringPoints };
  },
};

export default ArcheryCounter;
