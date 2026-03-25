import * as path from 'path';
import { loadImageNode } from '../src/imageLoader';
import { findTarget } from '../src/targetDetection';
import { SplineRing } from '../src/spline';

function ringCenter(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

function ringRadius(ring: SplineRing, cx: number, cy: number): number {
  return Math.max(...ring.points.map(p => Math.hypot(p[0] - cx, p[1] - cy)));
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn=r/255,gn=g/255,bn=b/255;
  const mx=Math.max(rn,gn,bn),mn=Math.min(rn,gn,bn),d=mx-mn;
  let h=0;
  if(d>0){
    if(mx===rn) h=60*(((gn-bn)/d)%6);
    else if(mx===gn) h=60*((bn-rn)/d+2);
    else h=60*((rn-gn)/d+4);
    if(h<0)h+=360;
  }
  return [h, mx>0?d/mx:0, mx];
}

function colorLabel(r:number,g:number,b:number):string {
  const [h,s,v]=rgbToHsv(r,g,b);
  if(s<0.15) return v>0.6 ? 'WHITE  ' : 'BLACK  ';
  if(h<18||h>342) return 'RED    ';
  if(h<70)  return 'YELLOW ';
  if(h<130) return 'GREEN  ';
  if(h<245) return 'BLUE   ';
  return 'OTHER  ';
}

async function main() {
  const imgPath = path.resolve(__dirname, '../images/20190325_201217.jpg');
  const { rgba, width, height } = await loadImageNode(imgPath);
  const result = findTarget(rgba, width, height);

  console.log('Image: ' + width + 'x' + height);
  console.log('Rings:');
  for (let i = 0; i < result.rings.length; i++) {
    const ring = result.rings[i];
    const [cx, cy] = ringCenter(ring);
    const r = ringRadius(ring, cx, cy);
    const prevRing = i > 0 ? result.rings[i - 1] : null;
    const ratio = prevRing ? (r / ringRadius(prevRing, cx, cy)).toFixed(3) : '-   ';
    console.log('  ring['+i+']: radius='+r.toFixed(1)+' cx='+cx.toFixed(0)+' cy='+cy.toFixed(0)+' ratio='+ratio);
  }

  const [cx, cy] = ringCenter(result.rings[0]);
  console.log('\nRadial scan right (dx=1, dy=0) from cx='+cx.toFixed(0)+' cy='+cy.toFixed(0)+':');
  for (let d = 5; d <= 310; d += 5) {
    const x = Math.round(cx + d), y = Math.round(cy);
    if (x >= width || y >= height || x < 0 || y < 0) { console.log('  d='+d+': OUT'); break; }
    const i = (y * width + x) * 4;
    const col = colorLabel(rgba[i], rgba[i+1], rgba[i+2]);
    const ringIdx = result.rings.findIndex((ring, ri) => {
      const [rcx, rcy] = ringCenter(ring);
      const rr = ringRadius(ring, rcx, rcy);
      const prevRr = ri > 0 ? ringRadius(result.rings[ri - 1], rcx, rcy) : 0;
      return rr >= d && prevRr < d;
    });
    console.log('  d='+String(d).padStart(3)+': '+col+' ring='+ringIdx+' rgb=('+rgba[i]+','+rgba[i+1]+','+rgba[i+2]+')');
  }
}
main().catch(console.error);
