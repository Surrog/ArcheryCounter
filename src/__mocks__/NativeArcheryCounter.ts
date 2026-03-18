import type { SplineRing } from '../spline';

const mockRings: SplineRing[] = Array.from({ length: 10 }, (_, i) => ({
  points: Array.from({ length: 8 }, (__, k) => {
    const r = 50 + i * 40;
    const theta = (2 * Math.PI * k) / 8;
    return [500 + r * Math.cos(theta), 500 + r * Math.sin(theta)] as [number, number];
  }),
}));

const ArcheryCounter = {
  processImage: jest.fn().mockResolvedValue(mockRings),
};

export type { SplineRing as RingEllipse };
export default ArcheryCounter;
