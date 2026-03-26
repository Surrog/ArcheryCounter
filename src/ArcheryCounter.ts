import { findTarget } from './targetDetection';
import { findArrows } from './arrowDetection';
import { scoreArrowWithCheck } from './scoring';
import { decodeBase64Jpeg } from './imageLoader';
import type { EllipseData, TargetBoundary, ColourCalibration, Pixel } from './targetDetection';
import type { SplineRing } from './spline';
import type { ArrowDetection } from './arrowDetection';
import type { ScoredArrow } from './scoring';

export type { SplineRing as RingEllipse };
export type { SplineRing };
export type { EllipseData, Pixel, TargetBoundary, ColourCalibration };
export type { ArrowDetection, ScoredArrow };

export interface ProcessImageResult {
  rings: SplineRing[];
  paperBoundary?: TargetBoundary;
  calibration?: ColourCalibration;
  arrows: ScoredArrow[];
  /** Raw per-ray transition points for each ring (index 0 = innermost). */
  ringPoints?: Pixel[][];
}

const ArcheryCounter = {
  async processImage(imageUri: string, base64: string): Promise<ProcessImageResult> {
    const { rgba, width, height } = decodeBase64Jpeg(base64);
    const result = findTarget(rgba, width, height);
    if (!result.success) throw new Error(result.error ?? 'Detection failed');
    const arrows = findArrows(rgba, width, height, result);
    const scored = result.calibration
      ? arrows.map(a => scoreArrowWithCheck(rgba, width, height, a, result.rings, result.calibration!))
      : arrows.map(a => ({ ...a, score: 0 as number | 'X' }));
    return { rings: result.rings, paperBoundary: result.paperBoundary, calibration: result.calibration, arrows: scored, ringPoints: result.ringPoints };
  },
};

export default ArcheryCounter;
