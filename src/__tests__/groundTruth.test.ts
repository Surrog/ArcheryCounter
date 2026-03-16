import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const ANNOTATIONS_PATH = path.resolve(__dirname, '../../images/annotate.json');
const IMAGES_DIR = path.resolve(__dirname, '../../images');

interface SplineRing {
  points: [number, number][];
}

interface ImageAnnotation {
  paperBoundary: [number, number][] | null;
  rings: SplineRing[];
}

interface AnnotationsFile {
  [filename: string]: ImageAnnotation;
}

function splineCentroid(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  const cx = ring.points.reduce((s, p) => s + p[0], 0) / n;
  const cy = ring.points.reduce((s, p) => s + p[1], 0) / n;
  return [cx, cy];
}

function splineRadius(ring: SplineRing): number {
  const [cx, cy] = splineCentroid(ring);
  const radii = ring.points.map(([x, y]) => Math.hypot(x - cx, y - cy));
  return radii.reduce((s, r) => s + r, 0) / radii.length;
}

// Try to load annotate.json
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
  annotations = null;
}

if (!annotations) {
  test('No annotations yet — run `npm run annotate`, annotate images, and export to images/annotate.json', () => {
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
        const [annCx, annCy] = splineCentroid(ann.rings[0]);
        const resCenter = result.rings[0];
        const centerDist = Math.hypot(resCenter.centerX - annCx, resCenter.centerY - annCy);
        expect(centerDist).toBeLessThan(25);
      }

      // Ring radius within 15% for each ring
      const minLen = Math.min(ann.rings.length, result.rings.length);
      for (let i = 0; i < minLen; i++) {
        const annRadius = splineRadius(ann.rings[i]);
        const resRadius = result.rings[i].width / 2;
        const relError = Math.abs(resRadius - annRadius) / annRadius;
        expect(relError).toBeLessThan(0.15);
      }
    },
    120000,
  );
}
