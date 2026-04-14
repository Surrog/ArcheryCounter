import { useCallback, useEffect, useRef, useState } from 'react';
import { launchImageLibrary } from 'react-native-image-picker';
import ArcheryCounter, { ScoredArrow } from './ArcheryCounter';
import type { TargetResult } from './ArcheryCounter';
import type { Pixel } from './targetDetection';

export interface ScorerState {
  imageUri: string | null;
  targets: TargetResult[] | null;
  arrows: ScoredArrow[] | null;
  ringPoints: Pixel[][] | null;
  /** Original pixel dimensions of the image, as reported by the image picker */
  imageWidth: number | null;
  imageHeight: number | null;
  loading: boolean;
  error: string | null;
}

const initialState: ScorerState = {
  imageUri: null,
  targets: null,
  arrows: null,
  ringPoints: null,
  imageWidth: null,
  imageHeight: null,
  loading: false,
  error: null,
};

export function useArcheryScorer() {
  const [state, setState] = useState<ScorerState>(initialState);
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

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
      const { targets, arrows, ringPoints = null } = await ArcheryCounter.processImage(uri, asset.base64!);
      if (mountedRef.current) {
        setState({ imageUri: uri, targets, arrows, ringPoints, imageWidth, imageHeight, loading: false, error: null });
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      if (mountedRef.current) setState({ ...initialState, error: message });
    }
  }, []);

  const reset = useCallback(() => setState(initialState), []);

  return { ...state, pickAndProcess, reset };
}
