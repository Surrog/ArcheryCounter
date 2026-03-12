import { NativeModules } from 'react-native';

export interface RingEllipse {
  /** X coordinate of the ellipse center in original image pixels */
  centerX: number;
  /** Y coordinate of the ellipse center in original image pixels */
  centerY: number;
  /** Full bounding-box width — semi-axis rx = width / 2 */
  width: number;
  /** Full bounding-box height — semi-axis ry = height / 2 */
  height: number;
  /** Rotation of the major axis in degrees (same as cv::RotatedRect::angle) */
  angle: number;
}

interface ArcheryCounterNativeInterface {
  processImage(imageUri: string): Promise<RingEllipse[]>;
}

const { ArcheryCounter } = NativeModules;

if (!ArcheryCounter) {
  console.warn(
    '[ArcheryCounter] Native module not found.\n' +
      'iOS: run `pod install` then rebuild.\n' +
      'Android: rebuild with CMake and register ArcheryCounterPackage in MainApplication.',
  );
}

export default ArcheryCounter as ArcheryCounterNativeInterface;
