#!/usr/bin/env tsx
import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../src/imageLoader';
import { findTarget } from '../src/targetDetection';
import { splineRadius } from '../src/spline';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const imgs = fs.readdirSync(IMAGES_DIR).filter(f => /\.jpg$/i.test(f)).sort();

async function main() {
  for (const img of imgs) {
    const { rgba, width, height } = await loadImageNode(path.join(IMAGES_DIR, img));
    const result = findTarget(rgba, width, height);
    if (!result.success || result.rings.length < 10) { console.log(`${img}: FAIL`); continue; }
    const r5 = splineRadius(result.rings[5]);
    const r7 = splineRadius(result.rings[7]);
    const ratio = r5 > 0 ? r7/r5 : 0;
    const flag = ratio < 1.05 || ratio > 1.55 ? '<<' : '';
    console.log(`${img.padEnd(35)} r5=${r5.toFixed(1)} r7=${r7.toFixed(1)} ratio=${ratio.toFixed(3)} ${flag}`);
  }
}
main().catch(console.error);
