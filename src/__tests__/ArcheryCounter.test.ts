import ArcheryCounter from '../ArcheryCounter';

jest.mock('../targetDetection', () => ({ findTarget: jest.fn() }));
jest.mock('../arrowDetection',  () => ({ findArrows:  jest.fn() }));
jest.mock('../imageLoader',     () => ({ decodeBase64Jpeg: jest.fn() }));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { findTarget }      = require('../targetDetection') as { findTarget: jest.Mock };
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { findArrows }      = require('../arrowDetection')  as { findArrows:  jest.Mock };
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { decodeBase64Jpeg } = require('../imageLoader')     as { decodeBase64Jpeg: jest.Mock };

const MOCK_RGBA   = new Uint8Array(4);
const MOCK_W      = 100;
const MOCK_H      = 100;
const MOCK_RINGS  = [{ points: [[50, 50], [60, 50], [60, 60], [50, 60]] as [number, number][] }];
const MOCK_BOUNDARY = { points: [[0, 0], [100, 0], [100, 100], [0, 100]] as [number, number][] };
const MOCK_ARROWS = [{ tip: [50, 50] as [number, number], nock: [100, 100] as [number, number] }];

describe('ArcheryCounter.processImage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    decodeBase64Jpeg.mockReturnValue({ rgba: MOCK_RGBA, width: MOCK_W, height: MOCK_H });
  });

  it('decodes the base64, runs detection, returns rings and arrows', async () => {
    findTarget.mockReturnValue({
      success: true, rings: MOCK_RINGS, paperBoundary: MOCK_BOUNDARY,
      calibration: null, ringPoints: null,
    });
    findArrows.mockReturnValue(MOCK_ARROWS);

    const result = await ArcheryCounter.processImage('file:///test.jpg', 'base64data');

    expect(decodeBase64Jpeg).toHaveBeenCalledWith('base64data');
    expect(findTarget).toHaveBeenCalledWith(MOCK_RGBA, MOCK_W, MOCK_H);
    expect(findArrows).toHaveBeenCalledWith(MOCK_RGBA, MOCK_W, MOCK_H, expect.objectContaining({ success: true }));
    expect(result.rings).toBe(MOCK_RINGS);
    expect(result.arrows).toBe(MOCK_ARROWS);
    expect(result.paperBoundary).toBe(MOCK_BOUNDARY);
  });

  it('throws the detection error when findTarget fails', async () => {
    findTarget.mockReturnValue({ success: false, rings: [], error: 'No colour blobs found' });

    await expect(ArcheryCounter.processImage('file:///bad.jpg', 'base64'))
      .rejects.toThrow('No colour blobs found');
    expect(findArrows).not.toHaveBeenCalled();
  });

  it('throws "Detection failed" when findTarget fails without an error field', async () => {
    findTarget.mockReturnValue({ success: false, rings: [] });

    await expect(ArcheryCounter.processImage('file:///bad.jpg', 'base64'))
      .rejects.toThrow('Detection failed');
  });
});
