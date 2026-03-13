import { findTarget } from './targetDetection';
import { decodeBase64Jpeg } from './imageLoader';
import type { EllipseData } from './targetDetection';

export type { EllipseData as RingEllipse };

const ArcheryCounter = {
  async processImage(imageUri: string, base64: string): Promise<EllipseData[]> {
    const { rgba, width, height } = decodeBase64Jpeg(base64);
    const result = findTarget(rgba, width, height);
    if (!result.success) throw new Error(result.error ?? 'Detection failed');
    return result.rings;
  },
};

export default ArcheryCounter;
