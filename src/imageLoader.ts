export interface ImageBuffer {
  rgba: Uint8Array;
  width: number;
  height: number;
}

/**
 * Decode a base64 JPEG string to an RGBA pixel buffer.
 * Used in the React Native app (jpeg-js, works in Hermes).
 */
export function decodeBase64Jpeg(base64: string): ImageBuffer {
  // atob is available in React Native / Hermes
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const jpeg = require('jpeg-js') as typeof import('jpeg-js');
  const { data, width, height } = jpeg.decode(bytes);
  return { rgba: new Uint8Array(data.buffer), width, height };
}

/**
 * Load a JPEG from the filesystem for Node.js / Jest tests.
 * NOT imported in the React Native bundle (jimp is a devDependency).
 * Resizes to max 1200px on the longest side to keep test runtime manageable.
 */
export async function loadImageNode(filePath: string): Promise<ImageBuffer> {
  // Use require for CJS/Jest compatibility (jimp is a devDependency, Node-only)
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Jimp } = require('jimp') as typeof import('jimp');
  const img = await Jimp.read(filePath);
  img.scaleToFit({ w: 1200, h: 1200 });
  return {
    rgba: new Uint8Array(img.bitmap.data.buffer),
    width: img.bitmap.width,
    height: img.bitmap.height,
  };
}
