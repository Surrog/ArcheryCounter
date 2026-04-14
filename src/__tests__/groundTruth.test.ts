import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
import { Pool } from 'pg';
import { scoreArrow } from '../scoring';
import { sampleClosedSpline, splineRadius } from '../spline';

const IMAGES_DIR    = path.resolve(__dirname, '../../images');
const TSX_BIN       = path.resolve(__dirname, '../../node_modules/.bin/tsx');
const DETECT_WORKER = path.resolve(__dirname, '../../scripts/detect-worker.ts');
const CONCURRENCY   = Math.min(os.cpus().length, 4);

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

afterAll(() => db.end());

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SplineRing { points: [number, number][]; }

interface ArrowAnnotation {
  tip:   [number, number];
  nock:  [number, number] | null;
  score: number | 'X' | null;
}

interface ImageAnnotation {
  paperBoundary: [number, number][] | null;
  rings: SplineRing[];
  arrows: ArrowAnnotation[];
}

interface GeneratedData {
  rings:         SplineRing[];
  paperBoundary: [number, number][] | null;
  arrows:        { tip: [number, number]; nock: [number, number] | null }[];
}

// ---------------------------------------------------------------------------
// IoU helpers
// ---------------------------------------------------------------------------

function signedArea(poly: [number, number][]): number {
  let area = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    area += x1 * y2 - x2 * y1;
  }
  return area / 2;
}

function clipPolygon(
  subject: [number, number][],
  clip: [number, number][],
): [number, number][] {
  let output = subject.slice();
  for (let i = 0; i < clip.length && output.length > 0; i++) {
    const input = output;
    output = [];
    const [ax, ay] = clip[i];
    const [bx, by] = clip[(i + 1) % clip.length];
    const inside = (px: number, py: number) =>
      (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0;
    const intersect = ([p1x, p1y]: [number, number], [p2x, p2y]: [number, number]): [number, number] => {
      const dx1 = p2x - p1x, dy1 = p2y - p1y;
      const dx2 = bx - ax,   dy2 = by - ay;
      const t = ((ax - p1x) * dy2 - (ay - p1y) * dx2) / (dx1 * dy2 - dy1 * dx2);
      return [p1x + t * dx1, p1y + t * dy1];
    };
    for (let j = 0; j < input.length; j++) {
      const curr = input[j];
      const prev = input[(j + input.length - 1) % input.length];
      const currIn = inside(curr[0], curr[1]);
      const prevIn = inside(prev[0], prev[1]);
      if (currIn) {
        if (!prevIn) output.push(intersect(prev, curr));
        output.push(curr);
      } else if (prevIn) {
        output.push(intersect(prev, curr));
      }
    }
  }
  return output;
}

function polyIoU(a: [number, number][], b: [number, number][]): number {
  const aCCW = signedArea(a) < 0 ? [...a].reverse() : a;
  const bCCW = signedArea(b) < 0 ? [...b].reverse() : b;
  const inter = clipPolygon(aCCW, bCCW);
  if (inter.length < 3) return 0;
  const interArea = Math.abs(signedArea(inter));
  const aArea = Math.abs(signedArea(aCCW));
  const bArea = Math.abs(signedArea(bCCW));
  const union = aArea + bArea - interArea;
  return union <= 0 ? 0 : interArea / union;
}

function pointToSplineDist(pt: [number, number], ring: SplineRing): number {
  const samples = sampleClosedSpline(ring.points, 60);
  let min = Infinity;
  for (const [sx, sy] of samples) {
    const d = Math.hypot(pt[0] - sx, pt[1] - sy);
    if (d < min) min = d;
  }
  return min;
}

// ---------------------------------------------------------------------------
// Algorithm hash (must match annotate.ts)
// ---------------------------------------------------------------------------

function computeAlgorithmHash(): string {
  const files = [
    path.resolve(__dirname, '../../src/targetDetection.ts'),
  ].filter(f => fs.existsSync(f)).map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

// ---------------------------------------------------------------------------
// Worker runner
// ---------------------------------------------------------------------------

function runWorker(imgPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn(TSX_BIN, [DETECT_WORKER, imgPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '', stderr = '';
    proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });
    const timer = setTimeout(() => {
      try { proc.kill('SIGTERM'); } catch {}
      reject(new Error('Worker timed out'));
    }, 5 * 60 * 1000);
    proc.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(`Worker exited ${code}: ${stderr.slice(0, 300)}`)); return; }
      try { resolve(JSON.parse(stdout)); }
      catch { reject(new Error(`Bad JSON from worker: ${stdout.slice(0, 200)}`)); }
    });
  });
}

// ---------------------------------------------------------------------------
// beforeAll: populate generated table using a process pool
// ---------------------------------------------------------------------------

beforeAll(async () => {
  await db.query(`
    CREATE TABLE IF NOT EXISTS generated (
      filename       TEXT PRIMARY KEY,
      algorithm_hash TEXT NOT NULL,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      arrows         JSONB NOT NULL DEFAULT '[]',
      width          INT,
      height         INT,
      updated_at     TIMESTAMPTZ DEFAULT NOW()
    )
  `);
  await db.query(`ALTER TABLE generated ADD COLUMN IF NOT EXISTS width INT`);
  await db.query(`ALTER TABLE generated ADD COLUMN IF NOT EXISTS height INT`);

  const currentHash = computeAlgorithmHash();
  const { rows } = await db.query('SELECT filename, algorithm_hash FROM generated');
  const inGenerated = new Map<string, string>(
    rows.map((r: any) => [r.filename as string, r.algorithm_hash as string]),
  );

  const imageFiles = fs
    .readdirSync(IMAGES_DIR)
    .filter(f => /\.(jpg|jpeg)$/i.test(f))
    .sort();

  const stale = imageFiles.filter(f => inGenerated.get(f) !== currentHash);

  if (stale.length === 0) {
    console.log('All images up to date in generated table.');
    return;
  }

  console.log(`Detecting ${stale.length} image(s) with ${CONCURRENCY} concurrent workers…`);
  let done = 0;

  for (let i = 0; i < stale.length; i += CONCURRENCY) {
    const batch = stale.slice(i, i + CONCURRENCY);
    await Promise.all(batch.map(async filename => {
      const imgPath = path.join(IMAGES_DIR, filename);
      try {
        const result = await runWorker(imgPath);
        await db.query(
          `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           ON CONFLICT (filename) DO UPDATE
             SET algorithm_hash = EXCLUDED.algorithm_hash,
                 paper_boundary = EXCLUDED.paper_boundary,
                 rings          = EXCLUDED.rings,
                 arrows         = EXCLUDED.arrows,
                 width          = EXCLUDED.width,
                 height         = EXCLUDED.height,
                 updated_at     = NOW()`,
          [filename, currentHash,
           JSON.stringify(result.paperBoundary),
           JSON.stringify(result.rings),
           JSON.stringify(result.arrows),
           result.width ?? null,
           result.height ?? null],
        );
        done++;
        console.log(`  [${done}/${stale.length}] ${filename}`);
      } catch (err) {
        console.error(`  FAILED ${filename}: ${err}`);
      }
    }));
  }
}, 30 * 60 * 1000);

// ---------------------------------------------------------------------------
// Tests: compare generated table against annotations table
// ---------------------------------------------------------------------------

const imageFiles = fs
  .readdirSync(IMAGES_DIR)
  .filter(f => /\.(jpg|jpeg)$/i.test(f))
  .sort();

test.each(imageFiles)(
  'ground truth: %s',
  async (filename) => {
    const [annResult, genResult] = await Promise.all([
      db.query('SELECT paper_boundary, rings, arrows FROM annotations WHERE filename = $1', [filename]),
      db.query('SELECT paper_boundary, rings, arrows FROM generated WHERE filename = $1', [filename]),
    ]);

    if (annResult.rows.length === 0) return; // no annotation — skip

    // Annotations table uses multi-target format: rings[target][ringSet][ring],
    // paper_boundary[target][point]. Unwrap the first target's data.
    const rawPb   = annResult.rows[0].paper_boundary;
    const rawRings = annResult.rows[0].rings ?? [];
    const ann: ImageAnnotation = {
      paperBoundary: Array.isArray(rawPb?.[0]?.[0]) ? rawPb[0] : rawPb,
      rings:         Array.isArray(rawRings[0]?.[0]) ? rawRings[0][0] : rawRings,
      arrows:        annResult.rows[0].arrows ?? [],
    };

    expect(genResult.rows.length).toBeGreaterThan(0);
    const gen: GeneratedData = {
      rings:         genResult.rows[0].rings ?? [],
      paperBoundary: genResult.rows[0].paper_boundary ?? null,
      arrows:        genResult.rows[0].arrows ?? [],
    };

    // Detection must succeed (rings always non-empty on success)
    expect(gen.rings.length).toBeGreaterThan(0);

    // Paper boundary IoU
    if (ann.paperBoundary && gen.paperBoundary) {
      const iou = polyIoU(ann.paperBoundary, gen.paperBoundary);
      if (iou < 0.5) console.error(`[${filename}] paper boundary IoU=${iou.toFixed(3)}`);
      expect(iou).toBeGreaterThan(0.25);
    }

    // Ring IoU: greedy bipartite matching (best IoU first) to correctly pair rings
    // despite sort-order misalignment caused by partially-wrong detected rings.
    if (ann.rings.length > 0 && gen.rings.length > 0) {
      const annValid = ann.rings.filter(r => splineRadius(r) >= 1);
      const annPolys = annValid.map(r => sampleClosedSpline(r.points, 60));
      const genPolys = gen.rings.map(r => sampleClosedSpline(r.points, 60));

      const pairs: { ai: number; gi: number; iou: number }[] = [];
      for (let ai = 0; ai < annPolys.length; ai++) {
        for (let gi = 0; gi < genPolys.length; gi++) {
          pairs.push({ ai, gi, iou: polyIoU(annPolys[ai], genPolys[gi]) });
        }
      }
      pairs.sort((a, b) => b.iou - a.iou);

      const matchedA = new Set<number>(), matchedG = new Set<number>();
      const ringIouFailures: number[] = [];
      for (const { ai, gi, iou } of pairs) {
        if (matchedA.has(ai) || matchedG.has(gi)) continue;
        matchedA.add(ai); matchedG.add(gi);
        if (iou < 0.5) console.error(`[${filename}] ring[${ai}↔${gi}] IoU=${iou.toFixed(3)}`);
        if (iou < 0.2) ringIouFailures.push(iou);
      }
      if (ringIouFailures.length > 7) {
        console.error(`[${filename}] ${ringIouFailures.length} rings have IoU < 0.2 (> 7 allowed)`);
      }
      expect(ringIouFailures.length).toBeLessThanOrEqual(7);
    }

    // Arrow detection assertions
    if (ann.arrows.length > 0) {
      const detected = gen.arrows;

      const fmtPt = (p: [number, number] | null) =>
        p ? `(${Math.round(p[0])},${Math.round(p[1])})` : 'null';
      const detSummary = () =>
        detected.map((d, i) => `  det[${i}] tip=${fmtPt(d.tip)} nock=${fmtPt(d.nock)}`).join('\n');
      const annSummary = () =>
        ann.arrows.map((a, i) => `  ann[${i}] tip=${fmtPt(a.tip)} nock=${fmtPt(a.nock)} score=${a.score}`).join('\n');

      // Count: must not miss more than 4 annotated arrows (allow unlimited extras —
      // images may contain arrows from previous rounds not covered by annotations).
      if (detected.length < ann.arrows.length - 4) {
        console.error(
          `[${filename}] missing arrows: detected ${detected.length}, expected ${ann.arrows.length}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(detected.length).toBeGreaterThanOrEqual(ann.arrows.length - 4);

      // Bijective tip matching
      const TIP_MATCH_PX = 45;
      type Pair = { ai: number; di: number; dist: number };
      const allPairs: Pair[] = [];
      for (let ai = 0; ai < ann.arrows.length; ai++) {
        for (let di = 0; di < detected.length; di++) {
          allPairs.push({ ai, di, dist: Math.hypot(
            detected[di].tip[0] - ann.arrows[ai].tip[0],
            detected[di].tip[1] - ann.arrows[ai].tip[1],
          )});
        }
      }
      allPairs.sort((a, b) => a.dist - b.dist);
      const matchedA = new Set<number>(), matchedD = new Set<number>();
      const tipMatchDist = new Map<number, number>();
      const tipMatchIdx  = new Map<number, number>();
      for (const { ai, di, dist } of allPairs) {
        if (matchedA.has(ai) || matchedD.has(di)) continue;
        if (dist <= TIP_MATCH_PX) {
          matchedA.add(ai); matchedD.add(di);
          tipMatchDist.set(ai, dist);
          tipMatchIdx.set(ai, di);
        }
      }
      const tipFailures: string[] = [];
      for (let ai = 0; ai < ann.arrows.length; ai++) {
        if (!tipMatchDist.has(ai)) {
          const a = ann.arrows[ai];
          let bestDist = Infinity, bestIdx = -1;
          for (let di = 0; di < detected.length; di++) {
            const d = Math.hypot(detected[di].tip[0] - a.tip[0], detected[di].tip[1] - a.tip[1]);
            if (d < bestDist) { bestDist = d; bestIdx = di; }
          }
          tipFailures.push(
            `  ann tip=${fmtPt(a.tip)} score=${a.score}` +
            ` → best det=${fmtPt(detected[bestIdx]?.tip ?? null)} dist=${bestDist.toFixed(1)}px`,
          );
        }
      }
      const maxTipFailures = Math.max(0, ann.arrows.length - detected.length) + 5;
      if (tipFailures.length > maxTipFailures) {
        console.error(
          `[${filename}] ${tipFailures.length} tip(s) unmatched (>${maxTipFailures} allowed):\n${tipFailures.join('\n')}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(tipFailures.length).toBeLessThanOrEqual(maxTipFailures);

      // Scoring assertions: use generated rings for geometry
      const scoreFailures: string[] = [];
      for (const [ai, di] of tipMatchIdx) {
        const annScore = ann.arrows[ai].score;
        if (annScore === null) continue;
        const detScore = scoreArrow(detected[di].tip, gen.rings);
        const numAnn = annScore === 'X' ? 10 : annScore;
        const numDet = detScore === 'X' ? 10 : detScore;
        const nearBoundary = gen.rings.some(r => pointToSplineDist(detected[di].tip, r) < 20);
        const ok = numDet === numAnn || (nearBoundary && Math.abs(numDet - numAnn) <= 1);
        if (!ok) {
          scoreFailures.push(
            `  ann[${ai}] tip=${fmtPt(ann.arrows[ai].tip)} score=${annScore} → det score=${detScore}` +
            (nearBoundary ? ' (near boundary)' : ''),
          );
        }
      }
      if (scoreFailures.length > 0) {
        console.error(`[${filename}] scoring failures:\n${scoreFailures.join('\n')}`);
      }
      expect(scoreFailures.length).toBeLessThanOrEqual(2);

      // Nock matching: informational only
      const matchedDet2 = new Set<number>();
      for (const annArrow of ann.arrows) {
        let bestDist = Infinity, bestIdx = -1;
        for (let di = 0; di < detected.length; di++) {
          if (matchedDet2.has(di)) continue;
          const d = Math.hypot(detected[di].tip[0] - annArrow.tip[0], detected[di].tip[1] - annArrow.tip[1]);
          if (d < bestDist) { bestDist = d; bestIdx = di; }
        }
        if (bestIdx >= 0) {
          matchedDet2.add(bestIdx);
        }
      }
    }
  },
  15000,
);
