import * as path from 'path';
import { loadImageNode } from '../src/imageLoader';
import { findTarget } from '../src/targetDetection';

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
  for(let i=0;i<result.rings.length;i++){
    const r=result.rings[i];
    const ratio = i>0 ? (r.width/result.rings[i-1].width).toFixed(3) : '-   ';
    console.log('  ring['+i+']: w='+r.width.toFixed(1)+' h='+r.height.toFixed(1)+' cx='+r.centerX.toFixed(0)+' cy='+r.centerY.toFixed(0)+' ratio='+ratio);
  }

  const cx=result.rings[0].centerX, cy=result.rings[0].centerY;
  console.log('\nRadial scan right (dx=1, dy=0) from cx='+cx.toFixed(0)+' cy='+cy.toFixed(0)+':');
  for(let d=5; d<=310; d+=5) {
    const x=Math.round(cx+d), y=Math.round(cy);
    if(x>=width||y>=height||x<0||y<0) { console.log('  d='+d+': OUT'); break; }
    const i=(y*width+x)*4;
    const col=colorLabel(rgba[i],rgba[i+1],rgba[i+2]);
    const ringIdx = result.rings.findIndex((r,ri)=>r.width/2>=d && (ri===0||result.rings[ri-1].width/2<d));
    console.log('  d='+String(d).padStart(3)+': '+col+' ring='+ringIdx+' rgb=('+rgba[i]+','+rgba[i+1]+','+rgba[i+2]+')');
  }
}
main().catch(console.error);
