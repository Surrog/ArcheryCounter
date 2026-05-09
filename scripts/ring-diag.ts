#!/usr/bin/env tsx
/**
 * Diagnose ring detection quality for specific images.
 */
import { Pool } from 'pg';
import { sampleClosedSpline, splineRadius } from '../src/spline';

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

const TARGET = process.argv[2];

async function main() {
  const where = TARGET ? `WHERE filename = '${TARGET}'` : `WHERE filename IN ('IMG-20260327-WA0002.jpg','20210619_143937.jpg','20210711_151526.jpg')`;
  const { rows } = await db.query(
    `SELECT g.filename, g.rings as gen_rings, a.rings as ann_rings
     FROM generated g JOIN annotations a USING(filename) ${where}`
  );

  for (const row of rows) {
    const genRings: { points: [number,number][] }[] = row.gen_rings ?? [];
    const annRings: { points: [number,number][] }[] = row.ann_rings ?? [];

    // Compute center from innermost ring
    const center = (r: typeof genRings[0]) => {
      const n = r.points.length;
      return [r.points.reduce((s,p)=>s+p[0],0)/n, r.points.reduce((s,p)=>s+p[1],0)/n];
    };

    const [gCx, gCy] = center(genRings[0]);
    const [aCx, aCy] = center(annRings[0]);

    const radii = (rings: typeof genRings, cx: number, cy: number) =>
      rings.map(r => {
        const n = r.points.length;
        const rcx = r.points.reduce((s,p)=>s+p[0],0)/n;
        const rcy = r.points.reduce((s,p)=>s+p[1],0)/n;
        return r.points.reduce((s,p)=>s+Math.hypot(p[0]-rcx,p[1]-rcy),0)/n;
      });

    const gR = radii(genRings, gCx, gCy);
    const aR = radii(annRings, aCx, aCy);

    console.log(`\n${row.filename}:`);
    console.log(`  Gen center=(${Math.round(gCx)},${Math.round(gCy)})  Ann center=(${Math.round(aCx)},${Math.round(aCy)})`);
    console.log(`  Ring radii (gen vs ann):`)
    for (let i=0; i<10; i++) {
      const ratio = aR[i] > 0 ? gR[i]/aR[i] : 0;
      console.log(`    ring[${i}]: gen=${Math.round(gR[i])}  ann=${Math.round(aR[i])}  ratio=${ratio.toFixed(3)}`);
    }
  }

  await db.end();
}

main().catch(e => { console.error(e); process.exit(1); });
