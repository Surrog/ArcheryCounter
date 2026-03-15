import { useCallback, useState } from 'react';
import { launchImageLibrary } from 'react-native-image-picker';
import ArcheryCounter, { RingEllipse } from './ArcheryCounter';
import type { TargetBoundary } from './targetDetection';

export interface ScorerState {
  imageUri: string | null;
  rings: RingEllipse[] | null;
  paperBoundary: TargetBoundary | null;
  /** Original pixel dimensions of the image, as reported by the image picker */
  imageWidth: number | null;
  imageHeight: number | null;
  loading: boolean;
  error: string | null;
}

const initialState: ScorerState = {
  imageUri: null,
  rings: null,
  paperBoundary: null,
  imageWidth: null,
  imageHeight: null,
  loading: false,
  error: null,
};

export function useArcheryScorer() {
  const [state, setState] = useState<ScorerState>(initialState);

  const pickAndProcess = useCallback(async () => {
    const pickerResult = await launchImageLibrary({
      mediaType: 'photo',
      includeBase64: true,
      maxWidth: 1200,
      maxHeight: 1200,
    });

    if (pickerResult.didCancel || !pickerResult.assets?.[0]) return;

    const asset = pickerResult.assets[0];
    const uri = asset.uri!;
    const imageWidth = asset.width ?? null;
    const imageHeight = asset.height ?? null;

    setState({ ...initialState, loading: true });

    try {
      const { rings, paperBoundary = null } = await ArcheryCounter.processImage(uri, asset.base64!);
      setState({ imageUri: uri, rings, paperBoundary, imageWidth, imageHeight, loading: false, error: null });
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      setState({ ...initialState, error: message });
    }
  }, []);

  const reset = useCallback(() => setState(initialState), []);

  return { ...state, pickAndProcess, reset };
}
