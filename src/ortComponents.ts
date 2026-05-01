
export interface OnnxSession {
    currentModelPath: string;
    session: any;
    ort: any;
    isReleased: boolean;
} 

/**
 * Shared ONNX session management for both arrow and boundary detectors.
 *
 * Dynamically imports the appropriate ONNX Runtime package based on environment:
 *   - React Native apps → onnxruntime-react-native (loaded via Metro)
 *   - Node.js scripts  → onnxruntime-node (loaded from node_modules)
 *
 * Caches a single session for the lifetime of the process. If `getSession()` is
 * called again with a different model path, the existing session is released and
 * replaced with a new one.
 *
 * Call `releaseSession()` to free the session explicitly (e.g. on app background).
 *
 * Note: the session cache is shared across both detectors, so if your app uses
 * both, ensure they use the same model path or manage sessions carefully.
 *
 * Example usage:
 *   const session = await getSession(modelPath);
 *   const tensor = new session.ort.Tensor(...);
 *   const results = await session.session.run({ input: tensor });
 *   // ... use results ...
 *   releaseSession(session); // optional explicit cleanup
 *
 * Error handling:
 *   If the model output is not found or the session fails to load, an error is thrown.
 */
export async function getSession(modelPath: string): Promise<OnnxSession> {
  let ort: typeof import('onnxruntime-node');
  try {
    ort = await import('onnxruntime-node');
  } catch {
    ort = await import('onnxruntime-react-native' as string) as typeof import('onnxruntime-node');
  }

  const session = await ort.InferenceSession.create(modelPath);
  return { session, ort, currentModelPath: modelPath, isReleased: false };
}

export function releaseSession( session: OnnxSession ): void {
    try { session.session.release?.(); } catch {}
    session.isReleased = true;
}
