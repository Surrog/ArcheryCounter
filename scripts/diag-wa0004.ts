import { loadImageNode } from '../src/imageLoader';
import { findArrows } from '../src/arrowDetection';
import { findTarget } from '../src/targetDetection';

process.env.DEBUG_ARROWS = '1';

async function main() {
  const img = await loadImageNode('./images/IMG-20260327-WA0004.jpg');
  const result = findTarget(img.rgba, img.width, img.height);
  if (!result.success) { console.error('findTarget failed'); return; }
  const arrows = findArrows(img.rgba, img.width, img.height, result);
  console.log('Detected tips:', JSON.stringify(arrows.map((a: any) => a.tip)));
}
main().catch(console.error);
