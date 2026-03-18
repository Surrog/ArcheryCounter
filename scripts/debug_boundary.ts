import { loadImageNode } from '../src/imageLoader';
import { findTarget } from '../src/targetDetection';

const IMAGES = [
  '20190321_211008.jpg',
  '20190321_212022.jpg',
  '20190325_193820.jpg',
];

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const r1 = r/255, g1 = g/255, b1 = b/255;
  const max = Math.max(r1,g1,b1), min = Math.min(r1,g1,b1), d = max-min, v = max;
  const s = max===0 ? 0 : d/max;
  let h = 0;
  if (d>0) {
    if(max===r1) h=60*(((g1-b1)/d)%6);
    else if(max===g1) h=60*((b1-r1)/d+2);
    else h=60*((r1-g1)/d+4);
    if(h<0) h+=360;
  }
  return [h,s,v];
}

async function debugImage(filename: string) {
  const { rgba, width, height } = await loadImageNode(`./images/${filename}`);
  const result = findTarget(rgba, width, height);
  if (!result.success) { console.log(`${filename}: FAILED`); return; }

  const r0 = result.rings[0];
  const cx = r0.centerX, cy = r0.centerY;
  const w = r0.width / 2;
  const startRadius = Math.max(w * 4, 20);

  console.log(`\n=== ${filename} (${width}x${height}) ===`);
  console.log(`Centre: (${cx.toFixed(0)}, ${cy.toFixed(0)})  w~${w.toFixed(0)}  startR=${startRadius.toFixed(0)}`);

  if (result.paperBoundary) {
    const verts = result.paperBoundary.points;
    console.log(`Polygon (${verts.length} verts): ${verts.map(([x,y])=>`(${x},${y})`).join(' ')}`);
  }

  // Scan each direction, show stopping pixel HSV
  const N = 72;
  const suspicious: string[] = [];
  for (let i = 0; i < N; i++) {
    const theta = (i/N)*2*Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    let lastD = startRadius;
    let stopReason = 'edge';
    let stopH = 0, stopS = 0, stopV = 0;

    for (let d = Math.max(1, Math.round(startRadius)); ; d++) {
      const x = cx + d*cosT, y = cy + d*sinT;
      if (x<0||x>=width||y<0||y>=height) { lastD=d; break; }
      const pidx = (Math.round(y)*width+Math.round(x))*4;
      const [h,s,v] = rgbToHsv(rgba[pidx], rgba[pidx+1], rgba[pidx+2]);
      if (s>0.25 && h>=15 && h<=65 && v>0.15) {
        lastD=d; stopReason='hay'; stopH=h; stopS=s; stopV=v; break;
      }
      lastD=d;
    }

    const angleDeg = Math.round(theta * 180/Math.PI);
    const endX = Math.round(cx + lastD*cosT);
    const endY = Math.round(cy + lastD*sinT);

    // Flag: endpoint very close to centre compared to neighbours, or near image edge in suspicious way
    const atEdge = endX<=1 || endX>=width-1 || endY<=1 || endY>=height-1;
    const prevI = (i+N-1)%N;
    suspicious.push(`${String(angleDeg).padStart(3)}°  dist=${String(Math.round(lastD)).padStart(4)}  (${String(endX).padStart(4)},${String(endY).padStart(4)})  ${stopReason}${stopReason==='hay'?' h='+stopH.toFixed(0)+' s='+stopS.toFixed(2)+' v='+stopV.toFixed(2):'      '}${atEdge?' [EDGE]':''}`);
  }
  suspicious.forEach(s => console.log(' '+s));
}

async function main() {
  for (const img of IMAGES) await debugImage(img);
}
main().catch(console.error);
