#!/usr/bin/env tsx
/**
 * For each threshold, compute recall using greedy bipartite matching.
 * Shows how many more arrows would be matched at each threshold.
 */
import { Pool } from 'pg';

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

function computeTP(
  annArrows: { tip: [number, number] }[],
  genArrows: { tip: [number, number] }[],
  threshold: number,
): number {
  if (!annArrows.length || !genArrows.length) return 0;
  const allPairs: { ai: number; di: number; dist: number }[] = [];
  for (let ai = 0; ai < annArrows.length; ai++) {
    for (let di = 0; di < genArrows.length; di++) {
      allPairs.push({ ai, di, dist: Math.hypot(
        genArrows[di].tip[0] - annArrows[ai].tip[0],
        genArrows[di].tip[1] - annArrows[ai].tip[1],
      )});
    }
  }
  allPairs.sort((a, b) => a.dist - b.dist);
  const matchedA = new Set<number>(), matchedD = new Set<number>();
  let tp = 0;
  for (const { ai, di, dist } of allPairs) {
    if (matchedA.has(ai) || matchedD.has(di)) continue;
    if (dist <= threshold) { matchedA.add(ai); matchedD.add(di); tp++; }
  }
  return tp;
}

async function main() {
  const { rows } = await db.query(
    'SELECT a.filename, a.arrows as ann_arrows, g.arrows as gen_arrows ' +
    'FROM annotations a LEFT JOIN generated g USING(filename) ORDER BY a.filename'
  );

  const thresholds = [45, 55, 60, 65, 70, 75, 80, 90, 100, 120];
  let totalAnn = 0;
  const tpByThr = new Map<number, number>(thresholds.map(t => [t, 0]));

  for (const row of rows) {
    const annArrows: { tip: [number, number] }[] = row.ann_arrows ?? [];
    const genArrows: { tip: [number, number] }[] = row.gen_arrows ?? [];
    if (!annArrows.length) continue;
    totalAnn += annArrows.length;
    for (const thr of thresholds) {
      tpByThr.set(thr, (tpByThr.get(thr) ?? 0) + computeTP(annArrows, genArrows, thr));
    }
  }

  console.log('Recall vs tip-matching threshold:');
  for (const thr of thresholds) {
    const tp = tpByThr.get(thr) ?? 0;
    const recall = tp / totalAnn;
    console.log(`  ${thr}px: TP=${tp}/${totalAnn} recall=${recall.toFixed(3)}`);
  }

  await db.end();
}

main().catch(e => { console.error(e); process.exit(1); });
