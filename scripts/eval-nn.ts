/**
 * Neural-network evaluation — recall@45px vs ground-truth annotations.
 *
 * Usage:
 *   npx tsx scripts/eval-nn.ts [--model scripts/nn/arrow_detector_fp32.onnx] [--threshold 0.35]
 *
 * Reads ground truth from the `annotations` table, runs NN inference on each
 * image, computes per-image recall@45px using the same greedy bipartite match
 * as the Python train.py, and prints a summary table + overall recall.
 */

import * as path from 'path';
import * as fs   from 'fs';
import { Pool }  from 'pg';
import { Jimp }  from 'jimp';
import { detectArrowsNN } from '../src/arrowDetector';

const IMAGES_DIR    = path.resolve(__dirname, '../images');
const DEFAULT_MODEL = path.resolve(__dirname, 'nn/arrow_detector_fp32.onnx');
const THRESHOLD_PX  = 45;

// ── CLI args ─────────────────────────────────────────────────────────────────

const args        = process.argv.slice(2);
const modelIdx    = args.indexOf('--model');
const threshIdx   = args.indexOf('--threshold');
const MODEL_PATH  = modelIdx  >= 0 ? args[modelIdx  + 1] : DEFAULT_MODEL;
const HM_THRESH   = threshIdx >= 0 ? parseFloat(args[threshIdx + 1]) : 0.35;

// ── DB ────────────────────────────────────────────────────────────────────────

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

// ── metric ────────────────────────────────────────────────────────────────────

function recallAt(
  preds: [number, number][],
  gts:   [number, number][],
  thr:   number,
): { tp: number; fn: number; recall: number } {
  if (gts.length === 0) return { tp: 0, fn: 0, recall: 1 };
  if (preds.length === 0) return { tp: 0, fn: gts.length, recall: 0 };

  const pairs: [number, number, number][] = [];
  for (let gi = 0; gi < gts.length; gi++) {
    for (let pi = 0; pi < preds.length; pi++) {
      const d = Math.hypot(gts[gi][0] - preds[pi][0], gts[gi][1] - preds[pi][1]);
      if (d <= thr) pairs.push([d, gi, pi]);
    }
  }
  pairs.sort((a, b) => a[0] - b[0]);

  const matchedG = new Set<number>(), matchedP = new Set<number>();
  for (const [, gi, pi] of pairs) {
    if (!matchedG.has(gi) && !matchedP.has(pi)) {
      matchedG.add(gi); matchedP.add(pi);
    }
  }
  const tp = matchedG.size;
  return { tp, fn: gts.length - tp, recall: tp / gts.length };
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  if (!fs.existsSync(MODEL_PATH)) {
    console.error(`Model not found: ${MODEL_PATH}`);
    console.error('Run: cd scripts/nn && uv run export.py --checkpoint checkpoints/best.pt --out arrow_detector_fp32.onnx');
    process.exit(1);
  }

  const { rows } = await db.query<{ filename: string; arrows: any[] }>(`
    SELECT filename, arrows
    FROM   annotations
    WHERE  arrows IS NOT NULL
      AND  jsonb_array_length(arrows) > 0
    ORDER  BY filename
  `);

  console.log(`Evaluating ${rows.length} annotated images with model: ${path.basename(MODEL_PATH)}`);
  console.log(`Heatmap threshold: ${HM_THRESH}   Recall threshold: ${THRESHOLD_PX}px\n`);

  let totalTp = 0, totalFn = 0, totalPred = 0;

  const colW = [30, 6, 6, 6, 8];
  const header = ['filename', 'GT', 'TP', 'FN', 'recall'].map((h, i) => h.padEnd(colW[i])).join('  ');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const row of rows) {
    const imgPath = path.join(IMAGES_DIR, row.filename);
    if (!fs.existsSync(imgPath)) continue;

    // Load image (≤1200px, matching training pre-processing)
    const img  = await Jimp.read(imgPath);
    img.scaleToFit({ w: 1200, h: 1200 });
    const rgba = new Uint8Array(img.bitmap.data.buffer);
    const { width, height } = img.bitmap;

    // Ground-truth tips (in ≤1200px coords)
    const origW = img.bitmap.width;   // same — we scaled in place
    const arrows: any[] = typeof row.arrows === 'string' ? JSON.parse(row.arrows) : row.arrows;
    const gtTips: [number, number][] = arrows
      .filter(a => a.tip != null)
      .map(a => [a.tip[0] * (width / origW), a.tip[1] * (height / origW)] as [number, number]);

    // NN inference
    const detections = await detectArrowsNN(rgba, width, height, MODEL_PATH, HM_THRESH);
    const predTips   = detections.map(d => d.tip as [number, number]);

    const { tp, fn, recall } = recallAt(predTips, gtTips, THRESHOLD_PX);
    totalTp   += tp;
    totalFn   += fn;
    totalPred += predTips.length;

    const cols = [
      row.filename.padEnd(colW[0]),
      String(gtTips.length).padEnd(colW[1]),
      String(tp).padEnd(colW[2]),
      String(fn).padEnd(colW[3]),
      recall.toFixed(3).padEnd(colW[4]),
    ];
    console.log(cols.join('  '));
  }

  const totalGt  = totalTp + totalFn;
  const overall  = totalGt > 0 ? totalTp / totalGt : 0;
  const prec     = totalPred > 0 ? totalTp / totalPred : 0;

  console.log('-'.repeat(70));
  console.log(`Overall  GT=${totalGt}  TP=${totalTp}  FN=${totalFn}  Pred=${totalPred}`);
  console.log(`Recall@${THRESHOLD_PX}px = ${overall.toFixed(3)}   Precision = ${prec.toFixed(3)}`);

  await db.end();
}

main().catch(err => { console.error(err); process.exit(1); });
