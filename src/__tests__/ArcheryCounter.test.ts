import ArcheryCounter from '../ArcheryCounter';

jest.mock('../targetDetection', () => ({
  findTarget: jest.fn(),
  pointInPolygon: jest.fn().mockReturnValue(true),
}));
jest.mock('../arrowDetector',   () => ({ detectArrowsNN: jest.fn() }));
jest.mock('../imageLoader',     () => ({ decodeBase64Jpeg: jest.fn() }));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { findTarget }      = require('../targetDetection') as { findTarget: jest.Mock };
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { detectArrowsNN }  = require('../arrowDetector')   as { detectArrowsNN: jest.Mock };
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { decodeBase64Jpeg } = require('../imageLoader')     as { decodeBase64Jpeg: jest.Mock };

const MOCK_RGBA    = new Uint8Array(4);
const MOCK_W       = 100;
const MOCK_H       = 100;
const MOCK_RINGS   = [{ points: [[50, 50], [60, 50], [60, 60], [50, 60]] as [number, number][] }];
const MOCK_BOUNDARY = { points: [[0, 0], [100, 0], [100, 100], [0, 100]] as [number, number][] };
const MOCK_ARROWS  = [{ tip: [50, 50] as [number, number], score: 9 }];
const MODEL_PATH   = '/models/arrow_detector.onnx';

describe('ArcheryCounter.processImage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    decodeBase64Jpeg.mockReturnValue({ rgba: MOCK_RGBA, width: MOCK_W, height: MOCK_H });
    findTarget.mockReturnValue({
      success: true,
      targets: [{ rings: MOCK_RINGS, paperBoundary: MOCK_BOUNDARY }],
      rings: MOCK_RINGS,
      paperBoundary: MOCK_BOUNDARY,
      calibration: null, ringPoints: null,
    });
  });

  it('returns empty arrows when no modelPath is provided', async () => {
    const result = await ArcheryCounter.processImage('file:///test.jpg', 'base64data');

    expect(decodeBase64Jpeg).toHaveBeenCalledWith('base64data');
    expect(findTarget).toHaveBeenCalledWith(MOCK_RGBA, MOCK_W, MOCK_H);
    expect(detectArrowsNN).not.toHaveBeenCalled();
    expect(result.rings).toBe(MOCK_RINGS);
    expect(result.arrows).toEqual([]);
    expect(result.paperBoundary).toBe(MOCK_BOUNDARY);
  });

  it('calls detectArrowsNN and returns its arrows when modelPath is provided', async () => {
    detectArrowsNN.mockResolvedValue(MOCK_ARROWS);

    const result = await ArcheryCounter.processImage('file:///test.jpg', 'base64data', { modelPath: MODEL_PATH });

    expect(detectArrowsNN).toHaveBeenCalledWith(MOCK_RGBA, MOCK_W, MOCK_H, MODEL_PATH);
    expect(result.arrows).toStrictEqual(MOCK_ARROWS);
  });

  it('returns empty arrows when detectArrowsNN throws', async () => {
    detectArrowsNN.mockRejectedValue(new Error('ONNX load failed'));

    const result = await ArcheryCounter.processImage('file:///test.jpg', 'base64data', { modelPath: MODEL_PATH });

    expect(result.arrows).toEqual([]);
  });

  it('logs a warning via console.warn when detectArrowsNN throws', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    detectArrowsNN.mockRejectedValue(new Error('model missing output key'));

    await ArcheryCounter.processImage('file:///test.jpg', 'base64data', { modelPath: MODEL_PATH });

    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toMatch(/Arrow detection failed/i);
    warnSpy.mockRestore();
  });

  it('throws the detection error when findTarget fails', async () => {
    findTarget.mockReturnValue({ success: false, rings: [], error: 'No colour blobs found' });

    await expect(ArcheryCounter.processImage('file:///bad.jpg', 'base64'))
      .rejects.toThrow('No colour blobs found');
    expect(detectArrowsNN).not.toHaveBeenCalled();
  });

  it('throws "Detection failed" when findTarget fails without an error field', async () => {
    findTarget.mockReturnValue({ success: false, rings: [] });

    await expect(ArcheryCounter.processImage('file:///bad.jpg', 'base64'))
      .rejects.toThrow('Detection failed');
  });
});
