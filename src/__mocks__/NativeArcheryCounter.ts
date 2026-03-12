import type { RingEllipse } from '../NativeArcheryCounter';

const mockRings: RingEllipse[] = Array.from({ length: 10 }, (_, i) => ({
  centerX: 500,
  centerY: 500,
  width: 100 + i * 80,
  height: 100 + i * 80,
  angle: 0,
}));

const ArcheryCounter = {
  processImage: jest.fn().mockResolvedValue(mockRings),
};

export type { RingEllipse };
export default ArcheryCounter;
