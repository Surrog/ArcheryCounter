import { useCallback, useState } from 'react';
import { launchImageLibrary } from 'react-native-image-picker';
import ArcheryCounter, { RingEllipse } from './NativeArcheryCounter';

export interface ScorerState {
  imageUri: string | null;
  rings: RingEllipse[] | null;
  /** Original pixel dimensions of the image, as reported by the image picker */
  imageWidth: number | null;
  imageHeight: number | null;
  loading: boolean;
  error: string | null;
}

const initialState: ScorerState = {
  imageUri: null,
  rings: null,
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
      includeBase64: false,
    });

    if (pickerResult.didCancel || !pickerResult.assets?.[0]) return;

    const asset = pickerResult.assets[0];
    const uri = asset.uri!;
    const imageWidth = asset.width ?? null;
    const imageHeight = asset.height ?? null;

    setState({ ...initialState, loading: true });

    try {
      const rings = await ArcheryCounter.processImage(uri);
      setState({ imageUri: uri, rings, imageWidth, imageHeight, loading: false, error: null });
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      setState({ ...initialState, error: message });
    }
  }, []);

  const reset = useCallback(() => setState(initialState), []);

  return { ...state, pickAndProcess, reset };
}
