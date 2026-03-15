import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const ANNOTATIONS_PATH = path.resolve(__dirname, '../../annotations.json');
const IMAGES_DIR = path.resolve(__dirname, '../../images');

interface RingAnnotation {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  angle: number;
}

interface ImageAnnotation {
  paperBoundary: [number, number][] | null;
  rings: RingAnnotation[];
}

interface AnnotationsFile {
  [filename: string]: ImageAnnotation;
}

// Try to load annotations.json
let annotations: AnnotationsFile | null = null;
try {
  if (fs.existsSync(ANNOTATIONS_PATH)) {
    const raw = fs.readFileSync(ANNOTATIONS_PATH, 'utf8').trim();
    if (raw.length > 0) {
      const parsed = JSON.parse(raw);
      if (Object.keys(parsed).length > 0) {
        annotations = parsed as AnnotationsFile;
      }
    }
  }
} catch (e) {
  // If reading/parsing fails, treat as no annotations
  annotations = null;
}

if (!annotations) {
  test('No annotations yet — run `npm run annotate`, annotate images, and export annotations.json', () => {
    // Always passing placeholder
    expect(true).toBe(true);
  });
} else {
  const annotatedEntries = Object.entries(annotations) as [string, ImageAnnotation][];

  test.each(annotatedEntries)(
    'ground truth: %s',
    async (filename, ann) => {
      const imgPath = path.join(IMAGES_DIR, filename);

      const { rgba, width, height } = await loadImageNode(imgPath);
      const result = findTarget(rgba, width, height);

      expect(result.success).toBe(true);

      // Paper boundary check: each annotated corner within 60px of the nearest detected vertex
      if (ann.paperBoundary && result.paperBoundary) {
        for (const annCorner of ann.paperBoundary) {
          const minDist = Math.min(
            ...result.paperBoundary.points.map(([x, y]) =>
              Math.hypot(x - annCorner[0], y - annCorner[1])
            )
          );
          expect(minDist).toBeLessThan(60);
        }
      }

      // Center distance for ring[0] within 25px
      if (ann.rings.length > 0 && result.rings.length > 0) {
        const annCenter = ann.rings[0];
        const resCenter = result.rings[0];
        const centerDist = Math.hypot(
          resCenter.centerX - annCenter.centerX,
          resCenter.centerY - annCenter.centerY,
        );
        expect(centerDist).toBeLessThan(25);
      }

      // Ring width within 15% for each ring
      const minLen = Math.min(ann.rings.length, result.rings.length);
      for (let i = 0; i < minLen; i++) {
        const annWidth = ann.rings[i].width;
        const resWidth = result.rings[i].width;
        const relError = Math.abs(resWidth - annWidth) / annWidth;
        expect(relError).toBeLessThan(0.15);
      }
    },
    120000,
  );
}
