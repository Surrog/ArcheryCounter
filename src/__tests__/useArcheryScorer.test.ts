import { act, renderHook } from '@testing-library/react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import ArcheryCounter from '../ArcheryCounter';
import { useArcheryScorer } from '../useArcheryScorer';

jest.mock('react-native-image-picker', () => ({
  launchImageLibrary: jest.fn(),
}));

jest.mock('../ArcheryCounter', () => ({
  __esModule: true,
  default: {
    processImage: jest.fn().mockResolvedValue({
      rings: Array(10).fill({ points: Array.from({ length: 8 }, () => [500, 400] as [number, number]) }),
      paperBoundary: null,
    }),
  },
}));

const mockLaunch = launchImageLibrary as jest.Mock;

describe('useArcheryScorer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('starts with empty state', () => {
    const { result } = renderHook(() => useArcheryScorer());
    expect(result.current.imageUri).toBeNull();
    expect(result.current.rings).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('does nothing when picker is cancelled', async () => {
    mockLaunch.mockResolvedValue({ didCancel: true });
    const { result } = renderHook(() => useArcheryScorer());

    await act(async () => {
      await result.current.pickAndProcess();
    });

    expect(result.current.imageUri).toBeNull();
  });

  it('sets rings and imageUri on success', async () => {
    mockLaunch.mockResolvedValue({
      didCancel: false,
      assets: [{ uri: 'file:///test.jpg', width: 1000, height: 800 }],
    });

    const { result } = renderHook(() => useArcheryScorer());

    await act(async () => {
      await result.current.pickAndProcess();
    });

    expect(result.current.imageUri).toBe('file:///test.jpg');
    expect(result.current.rings).toHaveLength(10);
    expect(result.current.imageWidth).toBe(1000);
    expect(result.current.imageHeight).toBe(800);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('sets error state when processImage rejects', async () => {
    mockLaunch.mockResolvedValue({
      didCancel: false,
      assets: [{ uri: 'file:///bad.jpg', width: 100, height: 100 }],
    });

    // Override mock for this test only
    (ArcheryCounter.processImage as jest.Mock).mockRejectedValueOnce(new Error('CV error'));

    const { result } = renderHook(() => useArcheryScorer());

    await act(async () => {
      await result.current.pickAndProcess();
    });

    expect(result.current.error).toBe('CV error');
    expect(result.current.rings).toBeNull();
  });

  it('sets error state when processImage rejects with a non-Error value', async () => {
    mockLaunch.mockResolvedValue({
      didCancel: false,
      assets: [{ uri: 'file:///bad.jpg', width: 100, height: 100 }],
    });
    (ArcheryCounter.processImage as jest.Mock).mockRejectedValueOnce('plain string error');

    const { result } = renderHook(() => useArcheryScorer());
    await act(async () => { await result.current.pickAndProcess(); });

    expect(result.current.error).toBe('plain string error');
  });

  it('reset clears state', async () => {
    mockLaunch.mockResolvedValue({
      didCancel: false,
      assets: [{ uri: 'file:///test.jpg', width: 1000, height: 800 }],
    });

    const { result } = renderHook(() => useArcheryScorer());

    await act(async () => {
      await result.current.pickAndProcess();
    });

    act(() => {
      result.current.reset();
    });

    expect(result.current.imageUri).toBeNull();
    expect(result.current.rings).toBeNull();
  });
});
