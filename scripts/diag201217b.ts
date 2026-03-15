import * as path from 'path';
import { loadImageNode } from '../src/imageLoader';

// Inline the bootstrap logic to see what w actually is
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn=r/255,gn=g/255,bn=b/255;
  const mx=Math.max(rn,gn,bn),mn=Math.min(rn,gn,bn),d=mx-mn;
  let h=0;
  if(d>0){
    if(mx===rn) h=60*(((gn-bn)/d)%6); else if(mx===gn) h=60*((bn-rn)/d+2); else h=60*((rn-gn)/d+4);
    if(h<0)h+=360;
  }
  return [h, mx>0?d/mx:0, mx];
}

async function main() {
  const { rgba, width, height } = await loadImageNode(
    path.resolve(__dirname, '../images/20190325_201217.jpg')
  );
  const cx = 281, cy = 512;

  // Scan radially in all 360 directions, collect transition distances
  const transitions: number[] = [];
  for (let ai = 0; ai < 360; ai++) {
    const theta = (ai / 360) * 2 * Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    let prevV = -1;
    for (let d = 5; d <= 320; d += 2) {
      const x = Math.round(cx + d * cosT), y = Math.round(cy + d * sinT);
      if (x < 0 || x >= width || y < 0 || y >= height) break;
      const i = (y * width + x) * 4;
      const [, , v] = rgbToHsv(rgba[i], rgba[i+1], rgba[i+2]);
      if (prevV >= 0 && Math.abs(v - prevV) > 0.07) {
        transitions.push(d);
      }
      prevV = v;
    }
  }

  // Histogram of transitions
  const bins: Record<number, number> = {};
  for (const d of transitions) {
    const b = Math.round(d / 5) * 5;
    bins[b] = (bins[b] || 0) + 1;
  }
  console.log('Transition histogram (distance: count):');
  const keys = Object.keys(bins).map(Number).sort((a,b) => a-b);
  for (const k of keys) {
    if (bins[k] >= 20) console.log('  d=' + String(k).padStart(3) + ': ' + '█'.repeat(Math.min(60, bins[k])) + ' (' + bins[k] + ')');
  }
}
main().catch(console.error);
