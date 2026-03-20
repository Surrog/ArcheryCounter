import { findTarget } from './targetDetection';
import { findArrows } from './arrowDetection';
import { decodeBase64Jpeg } from './imageLoader';
import type { EllipseData, TargetBoundary, ColourCalibration, Pixel } from './targetDetection';
import type { SplineRing } from './spline';
import type { ArrowDetection } from './arrowDetection';

export type { SplineRing as RingEllipse };
export type { SplineRing };
export type { EllipseData, Pixel, TargetBoundary, ColourCalibration };
export type { ArrowDetection };

export interface ProcessImageResult {
  rings: SplineRing[];
  paperBoundary?: TargetBoundary;
  calibration?: ColourCalibration;
  arrows: ArrowDetection[];
}

const ArcheryCounter = {
  async processImage(imageUri: string, base64: string): Promise<ProcessImageResult> {
    const { rgba, width, height } = decodeBase64Jpeg(base64);
    const result = findTarget(rgba, width, height);
    if (!result.success) throw new Error(result.error ?? 'Detection failed');
    const arrows = findArrows(rgba, width, height, result);
    return { rings: result.rings, paperBoundary: result.paperBoundary, calibration: result.calibration, arrows };
  },
};

export default ArcheryCounter;
