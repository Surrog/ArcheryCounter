import * as fs from 'fs';
import { loadImageNode } from '../src/imageLoader';
import { findTarget } from '../src/targetDetection';

function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

const imgs = fs.readdirSync('images').filter((f: string) => f.endsWith('.jpg')).sort();

async function main() {
  for (const img of imgs) {
    const { rgba, width, height } = await loadImageNode('images/' + img);
    const r = findTarget(rgba, width, height);
    if (!r.success || !r.paperBoundary) continue;
    const cx = r.rings[0].centerX, cy = r.rings[0].centerY;

    const pts = r.paperBoundary.points;
    const n = pts.length;
    const maxBR   = Math.max(...pts.map(([bx, by]) => Math.hypot(bx - cx, by - cy)));
    const inradius = Math.min(...pts.map(([ax, ay], i) => {
      const [bx, by] = pts[(i + 1) % n];
      return distToSegment(cx, cy, ax, ay, bx, by);
    }));

    const violations = r.rings.map((ring, i) => ({ i, semi: ring.width / 2, ratio: ring.width / 2 / inradius }))
      .filter(x => x.ratio > 1.0);

    if (violations.length) {
      console.log(`${img}  inradius=${inradius.toFixed(1)} maxBR=${maxBR.toFixed(1)}`);
      violations.forEach(v => console.log(`  ring[${v.i}] semi=${v.semi.toFixed(1)} semi/inradius=${v.ratio.toFixed(3)}`));
    } else {
      const maxRatio = Math.max(...r.rings.map(ring => ring.width / 2 / inradius));
      console.log(`${img}  OK  max(semi/inradius)=${maxRatio.toFixed(3)}  inradius=${inradius.toFixed(1)} maxBR=${maxBR.toFixed(1)}`);
    }
  }
}
main().catch(console.error);
