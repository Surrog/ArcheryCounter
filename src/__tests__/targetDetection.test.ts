import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../../images');

const jpgFiles = fs.readdirSync(IMAGES_DIR)
  .filter(f => f.endsWith('.jpg'))
  .map(f => path.join(IMAGES_DIR, f));

describe('findTarget', () => {
  test.each(jpgFiles)('%s — detects 10 concentric rings', async (imgPath) => {
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);
    expect(result.rings).toHaveLength(10);

    for (const ring of result.rings) {
      expect(ring.width).toBeGreaterThan(0);
      expect(ring.height).toBeGreaterThan(0);
      expect(ring.centerX).toBeGreaterThanOrEqual(0);
      expect(ring.centerX).toBeLessThanOrEqual(width);
      expect(ring.centerY).toBeGreaterThanOrEqual(0);
      expect(ring.centerY).toBeLessThanOrEqual(height);
    }

    // All centers within 100 px of ring[0] center
    const { centerX: ox, centerY: oy } = result.rings[0];
    for (const ring of result.rings) {
      const d = Math.hypot(ring.centerX - ox, ring.centerY - oy);
      expect(d).toBeLessThanOrEqual(100);
    }
  }, 60000);
});
