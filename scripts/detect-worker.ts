/**
 * Standalone detection worker for AW-5 background generation.
 * Invoked by the annotation server as a child process:
 *   tsx detect-worker.ts <imgPath>
 *
 * Outputs a single JSON line to stdout (no base64 — avoids pipe overflow).
 * All other logging goes to stderr so the parent can distinguish them.
 */
import * as path from 'path';
import { loadImageNode } from '../src/imageLoader';
import { findTarget } from '../src/targetDetection';
import { findArrows } from '../src/arrowDetection';

const imgPath = process.argv[2];
if (!imgPath) {
  console.error('Usage: detect-worker.ts <imgPath>');
  process.exit(1);
}

(async () => {
  const { rgba, width, height } = await loadImageNode(imgPath);
  const result = findTarget(rgba, width, height);
  const arrows = findArrows(rgba, width, height, result);
  const output = {
    rings:         result.success ? result.rings : [],
    paperBoundary: result.success && result.paperBoundary ? result.paperBoundary.points : null,
    arrows,
    success:       result.success,
    error:         result.success ? undefined : (result as any).error,
  };
  process.stdout.write(JSON.stringify(output));
})().catch(err => {
  console.error('detect-worker error:', err);
  process.exit(1);
});
