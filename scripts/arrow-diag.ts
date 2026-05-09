#!/usr/bin/env tsx
/**
 * Per-image arrow detection diagnosis.
 */
import { Pool } from 'pg';

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

async function main() {
  const { rows } = await db.query(
    'SELECT a.filename, a.arrows as ann_arrows, g.arrows as gen_arrows ' +
    'FROM annotations a LEFT JOIN generated g USING(filename) ORDER BY a.filename'
  );

  const TIP_PX = 45;
  let totalTP = 0, totalFN = 0;

  for (const row of rows) {
    const filename: string = row.filename;
    const annArrows: { tip: [number, number]; nock: [number, number] | null; score: number | 'X' | null }[] = row.ann_arrows ?? [];
    const genArrows: { tip: [number, number] }[] = row.gen_arrows ?? [];

    if (annArrows.length === 0) continue;

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
    for (const { ai, di, dist } of allPairs) {
      if (matchedA.has(ai) || matchedD.has(di)) continue;
      if (dist <= TIP_PX) { matchedA.add(ai); matchedD.add(di); }
    }

    const tp = matchedA.size;
    const fn = annArrows.length - tp;
    totalTP += tp;
    totalFN += fn;

    // Print missed arrows
    const missed = annArrows
      .filter((_, ai) => !matchedA.has(ai))
      .map(a => `tip=(${Math.round(a.tip[0])},${Math.round(a.tip[1])}) score=${a.score}`);

    const status = fn > 0 ? `MISS ${fn}` : 'OK';
    const extra = genArrows.length - tp;
    console.log(
      `${status.padEnd(7)} ${filename.padEnd(35)} ann=${annArrows.length} det=${genArrows.length} tp=${tp} fn=${fn} extra=${extra}`
    );
    if (fn > 0) {
      for (const m of missed) console.log(`         missed: ${m}`);
    }
  }

  const total = totalTP + totalFN;
  console.log(`\nTotal: TP=${totalTP} FN=${totalFN} recall=${(totalTP/total).toFixed(3)} (${total} annotated arrows)`);

  await db.end();
}

main().catch(e => { console.error(e); process.exit(1); });
