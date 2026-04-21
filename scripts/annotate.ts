import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import * as crypto from 'crypto';
import { Pool } from 'pg';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, findRingSetFromCenter, ArcheryResult, TargetBoundary, isTargetBoundary } from '../src/targetDetection';
import { isSplineRing, splineCentroid, SplineRing, RingSet, isRingSet } from '../src/spline';
import { detectArrowsNN } from '../src/arrowDetector';
import { TargetData, ImageData, ArrowData, generateddbToTargets, annotationToTargets, logEvent, LOG_PATH } from './annotateInterface';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const PORT = parseInt(process.env.ANNOTATE_PORT || '3737', 10);

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

const RING_COLORS = [
  '#FFD700', '#FFD700',
  '#E8000D', '#E8000D',
  '#006CB7', '#006CB7',
  '#888888', '#888888',
  '#FFFFFF', '#FFFFFF',
];

// ---------------------------------------------------------------------------
// Logging (AW-6)
// ---------------------------------------------------------------------------

fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });

// ---------------------------------------------------------------------------
// Worker process for background detection (AW-5)
// ---------------------------------------------------------------------------

function clampBoundary(
  pts: [number, number][] | null,
  w: number,
  h: number,
): [number, number][] | null {
  if (!pts) return null;
  return pts.map(([x, y]) => [
    Math.round(Math.max(0, Math.min(w - 1, x))),
    Math.round(Math.max(0, Math.min(h - 1, y))),
  ]);
}

async function loadImage(imgPath: string, filename: string): Promise<ImageData> {
  let base64: string;
  let width: number;
  let height: number;

  console.log(`fast path: loading ${filename}…`);
  ({ base64, width, height } = await loadImageBase64(imgPath));
  return {
    filename: filename,
    base64: base64,
    height: height,
    width: width,
    generated: {
      arrows: [],
      targets: [],
    },
    annotated: {
      arrows: [],
      targets: [],
    },
  }
}

async function fetchGeneratedData(data: ImageData): Promise<[ImageData, boolean]>  {
  const { rows } = await db.query(
    'SELECT rings, paper_boundary, arrows FROM generated WHERE filename = $1',
    [data.filename],
  );
  if (rows.length == 0) {
    return [data, false]
  }
  if (rows.length > 1) {
    console.warn(`multiple rows for image : ${data.filename}`)
  } 

  data.generated.targets = generateddbToTargets(rows[0].paper_boundary, rows[0].rings);
  data.generated.arrows = rows[0].arrows ?? [];
  console.log(`found generated data: ${JSON.stringify(data.generated, null, 2)}`)

  return [data, true];
}

async function fetchAnnotationData(data: ImageData): Promise<[ImageData, boolean]>  {
  const { rows } = await db.query(
    'SELECT rings, paper_boundary, arrows FROM annotations WHERE filename = $1',
    [data.filename],
  );
  if (rows.length == 0) {
    return [data, false]
  }
  if (rows.length > 1) {
    console.warn(`multiple rows for image : ${data.filename}`)
  } 

  data.annotated.targets = annotationToTargets(rows[0].paper_boundary, rows[0].rings);
  data.annotated.arrows = rows[0].arrows ?? [];
  console.log(`found annotation data: ${JSON.stringify(data.annotated, null, 2)}`)

  return [data, true];
}


/** Convert TargetData[] → DB format { boundary: TargetBoundary[], rings: RingSet[], arrows: ArrowData[] } */
function targetsToDB(targets: TargetData[], arrows: ArrowData[]): { boundary: TargetBoundary[]; rings: RingSet[], arrows: ArrowData[] } {
  return {
    boundary: targets.map((t: TargetData) => t.paperBoundary ?? []),
    rings:    targets.map((t: TargetData) => t.ringSets ?? []).flat(),
    arrows:   arrows,
  };
}

/** Hash of algorithm source files — used to detect stale detections in DB. */
function computeAlgorithmHash(): string {
  const files = [
    path.resolve(__dirname, '../src/targetDetection.ts'),
  ].filter(f => fs.existsSync(f)).map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

async function loadImageBase64(imgPath: string): Promise<{ base64: string; width: number; height: number }> {
  const { Jimp } = require('jimp');
  const img = await Jimp.read(imgPath);
  img.scaleToFit({ w: 1200, h: 1200 });
  const base64 = await img.getBase64('image/jpeg');
  return { base64, width: img.width, height: img.height };
}

async function processImage(imgPath: string, imgData: ImageData): Promise<ImageData> {
  const filename = path.basename(imgPath);
  console.log(`  [1/3] loadImageNode ${filename}…`);
  const { rgba, width, height } = await loadImageNode(imgPath);
  console.log(`  [2/3] findTarget ${filename}…`);
  const result = findTarget(rgba, width, height);
  if (result.success) {
    console.log(`  findTarget success: ${filename}`);
    for (const t of result.targets) { 
      imgData.generated.targets.push({
        paperBoundary: t.paperBoundary,
        ringSets: [t.rings],
      });
    }
  }
  console.log(`  done: ${filename}`);
  console.log(`  [3/3] detectArrows ${filename}…`);
  const arrows = await detectArrowsNN(rgba, width, height, path.resolve(__dirname, '../models/arrow-detector.onnx'));
  for (const a of arrows) {
    imgData.generated.arrows.push({ tip: a.tip, score: a.score });
  }
  return imgData;
}


function generateHtml(filenames: string[]): string {
  const tpl = fs.readFileSync(path.join(__dirname, 'annotate.html'), 'utf-8');
  return tpl
    .replaceAll('{{IMAGES_TAG}}',      JSON.stringify(filenames))
    .replaceAll('{{RING_COLORS_TAG}}', JSON.stringify(RING_COLORS))
}

type GenState = 'ready' | 'queued' | 'computing' | 'error' | 'unknown';

interface SSEMessage {
  type: 'status' | 'new_image' | 'removed';
  filename: string;
  state?: GenState; 
}

async function main(): Promise<void> {
  const jpgFiles = fs
    .readdirSync(IMAGES_DIR)
    .filter(f => /\.(jpg|jpeg)$/i.test(f))
    .map(f => path.join(IMAGES_DIR, f))
    .sort();

  if (jpgFiles.length === 0) {
    console.error(`No JPEG files found in ${IMAGES_DIR}`);
    process.exit(1);
  }

  const filenames = jpgFiles.map(f => path.basename(f));
  const currentHash = computeAlgorithmHash();

  // --- DB setup (AW-1) ---
  await db.query(`
    CREATE TABLE IF NOT EXISTS public.annotations (
      filename text NOT NULL,
      paper_boundary jsonb NULL,
      rings jsonb DEFAULT '[]'::jsonb NOT NULL,
      updated_at timestamptz DEFAULT now() NOT NULL,
      arrows jsonb DEFAULT '[]'::jsonb NOT NULL,
      CONSTRAINT annotations_pkey PRIMARY KEY (filename)
    )
  `);
  await db.query(`
    CREATE TABLE IF NOT EXISTS public."generated" (
      filename text NOT NULL,
      algorithm_hash text NOT NULL,
      paper_boundary jsonb NULL,
      rings jsonb DEFAULT '[]'::jsonb NOT NULL,
      arrows jsonb DEFAULT '[]'::jsonb NOT NULL,
      updated_at timestamptz DEFAULT now() NOT NULL,
      width int4 NULL,
      height int4 NULL,
      CONSTRAINT generated_pkey PRIMARY KEY (filename)
    )
  `);

  // Wipe annotations rows with corrupt ring data
  const { rowCount: wiped } = await db.query(`
    UPDATE annotations SET rings = '[]', paper_boundary = NULL
     WHERE rings::text LIKE '%null%' AND rings <> 'null'::jsonb
  `);
  if (wiped) {
    console.log(`Wiped ${wiped} annotation rows with corrupt ring data.`);
    logEvent('warn', 'db_wipe', '', `${wiped} rows with null coordinates reset`);
  }

  console.log('Tables ready.');
  console.log(`Algorithm hash: ${currentHash}`);

  const { rows: genRows } = await db.query('SELECT filename, algorithm_hash FROM generated');
  const inGenerated = new Map<string, string>(genRows.map((r: any) => [r.filename, r.algorithm_hash as string]));

  const { rows: annRows } = await db.query('SELECT filename FROM annotations');
  const inAnnotations = new Set<string>(annRows.map((r: any) => r.filename as string));

  const readyCount = [...inGenerated.values()].filter(h => h === currentHash).length;
  const staleCount = filenames.length - readyCount;
  console.log(`Images: ${filenames.length} total — ${readyCount} ready, ${staleCount} stale/new`);

  const generationStatus = new Map<string, GenState>(
    filenames.map(f => [f, inGenerated.get(f) === currentHash ? 'ready' : 'queued']),
  );
  const sseClients = new Set<http.ServerResponse>();
  function broadcastSSE(data: SSEMessage) {
    generationStatus.set(data.filename, data.state ?? 'unknown');
    const msg = `data: ${JSON.stringify(data)}\n\n`;
    for (const client of sseClients) {
      try { client.write(msg); } catch { sseClients.delete(client); }
    }
  }

  const imageCache = new Map<string, ImageData>();
  // Deduplicates concurrent /api/image/ requests for the same file so detection
  // runs at most once.  Second caller awaits the first caller's promise.

  const html = generateHtml(filenames);

  async function computeGeneratedData(filename: string, imgPath: string, imageData: ImageData): Promise<ImageData> {
    broadcastSSE({ type: 'status', filename, state: 'computing' });
    imageData = await processImage(imgPath, imageData);
    const {boundary: dbBoundary, rings: dbRings, arrows: dbArrows} = targetsToDB(imageData.generated.targets, imageData.generated.arrows);
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
        JSON.stringify(dbBoundary), JSON.stringify(dbRings),
        JSON.stringify(dbArrows), imageData.width, imageData.height],
    );
    inGenerated.set(filename, currentHash);
    if (!inAnnotations.has(filename)) {
      await db.query(`INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (filename) DO UPDATE
                 SET paper_boundary = EXCLUDED.paper_boundary,
                     rings          = EXCLUDED.rings,
                     arrows         = EXCLUDED.arrows,
                     updated_at     = NOW()`, 
        [filename, 
          JSON.stringify(dbBoundary), 
          JSON.stringify(dbRings), 
          JSON.stringify(dbArrows)],);
      inAnnotations.add(filename);
    }

    broadcastSSE({ type: 'status', filename, state: 'ready' });
    return imageData;
  }

  // --- HTTP server ---
  const server = http.createServer(async (req, res) => {
    const respond = (status: number, body: string, type = 'application/json') => {
      res.writeHead(status, { 'Content-Type': type });
      res.end(body);
    };

    if (req.method === 'GET' && req.url === '/') {
      respond(200, html, 'text/html; charset=utf-8');

    } else if (req.method === 'GET' && req.url === '/api/stale-images') {
      const stale = filenames.filter(f => inGenerated.get(f) !== currentHash);
      respond(200, JSON.stringify({ stale }));

    } else if (req.method === 'GET' && req.url === '/api/generation-status') {
      const out: Record<string, string> = {};
      for (const [f, s] of generationStatus) out[f] = s;
      respond(200, JSON.stringify(out));

    } else if (req.method === 'GET' && req.url === '/api/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      res.write(`data: ${JSON.stringify({ type: 'connected' })}\n\n`);
      sseClients.add(res);
      req.on('close', () => sseClients.delete(res));

    } else if (req.method === 'GET' && req.url?.startsWith('/api/image-status/')) {
      const filename = decodeURIComponent(req.url.slice('/api/image-status/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      if (!inGenerated.has(filename)) { respond(200, '{"state":"new"}'); return; }
      if (inGenerated.get(filename) !== currentHash) { respond(200, '{"state":"stale"}'); return; }
      respond(200, '{"state":"ready"}');

    } else if (req.method === 'GET' && req.url?.startsWith('/api/image/')) {
      const filename = decodeURIComponent(req.url.slice('/api/image/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      try {
      // if the image is not in cache, load it (and run detection if needed) before responding
      if (!imageCache.has(filename)) {
        try {
          const imgPath = path.join(IMAGES_DIR, filename);
          if (!fs.existsSync(imgPath)) { respond(404, '{"error":"not found"}'); return; }
  
          const hashInGen = inGenerated.get(filename);
          let isReady: boolean = false;
  
          console.log(`image request: ${filename}  hashInGen=${hashInGen ?? 'null'}  isReady=${isReady}`);
  
          let imageData: ImageData = await loadImage(imgPath, filename)
          
          if (hashInGen === currentHash) {
            [imageData, isReady] = await fetchGeneratedData(imageData)
          }
  
          if (!isReady) {
            imageData = await computeGeneratedData(filename, imgPath, imageData)
          }

          [imageData, isReady] = await fetchAnnotationData(imageData);
          
          imageCache.set(filename, imageData)
        } catch (err) {
          console.error('Compute error:', err);
          logEvent('error', 'compute_failed', filename, String(err));
          broadcastSSE({ type: 'status', filename, state: 'error' });
          respond(500, '{"error":"internal error"}');
          return;
        }
      }

      respond(200, JSON.stringify(imageCache.get(filename)!));
    } catch (e) {
      console.error('Error processing image:', e);
      respond(500, '{"error":"internal error"}');
    } 
    } else if (req.method === 'POST' && req.url?.startsWith('/api/recompute/')) {
      const filename = decodeURIComponent(req.url.slice('/api/recompute/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      imageCache.delete(filename);
      inGenerated.delete(filename);
      inAnnotations.delete(filename);
      await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
      await db.query('DELETE FROM annotations WHERE filename = $1', [filename]);
      broadcastSSE({ type: 'status', filename, state: 'computing' });
      respond(202, '{"status":"computing"}');
      try {
        const imgPath = path.join(IMAGES_DIR, filename);
        let imageData = await computeGeneratedData(filename, imgPath, await loadImage(imgPath, filename));
        let isValid = true;
        [imageData, isValid] = await fetchAnnotationData(imageData);
        imageCache.set(filename, imageData)
      } catch (err) {
        console.error('Recompute error:', err);
        logEvent('error', 'recompute_failed', filename, String(err));
        broadcastSSE({ type: 'status', filename, state: 'error' });
      }

    } else if (req.method === 'POST' && req.url === '/api/save') {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      req.on('error', (err) => {
        console.error('Request error:', err);
        logEvent('error', 'save-request-error', '', String(err));
        respond(500, JSON.stringify({ error: String(err) }));
      });
      req.on('end', async () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          const received = Object.keys(data).length;
          console.log(`[save] received ${received} annotation(s)`);
          let saved = 0;
          for (const [filename, ann] of Object.entries(data) as [string, any][]) {
            const imgMeta = imageCache.get(filename);
            const w = imgMeta?.width ?? 0, h = imgMeta?.height ?? 0;
            const targets = (ann.targets ?? []).map((t: any) => ({
              paperBoundary:  { points: clampBoundary(t.paperBoundary.points, w, h) ?? [] },
              ringSets:      t.ringSets ?? [],
            }));
            const { boundary: dbBoundary, rings: dbRings, arrows: dbArrows } = targetsToDB(targets, ann.arrows ?? []);
            await db.query(
              `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (filename) DO UPDATE
                 SET paper_boundary = EXCLUDED.paper_boundary,
                     rings          = EXCLUDED.rings,
                     arrows         = EXCLUDED.arrows,
                     updated_at     = NOW()`,
              [filename, JSON.stringify(dbBoundary), JSON.stringify(dbRings), JSON.stringify(dbArrows)],
            );
            console.log(`[save]   OK ${filename}: targets=${targets.length}, arrows=${dbArrows.length}`);
            logEvent('info', 'save-ok', filename, `targets=${targets.length} arrows=${dbArrows.length}`);
            saved++;
          }
          console.log(`[save] done: saved=${saved}`);
          respond(200, JSON.stringify({ ok: true, saved }));
        } catch (e) {
          console.error('Save error:', e);
          logEvent('error', 'save-error', '', String(e));
          respond(500, JSON.stringify({ error: String(e) }));
        }
      });

    } else if (req.method === 'POST' && req.url === '/api/detect-ringset') {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      req.on('error', (err) => {
        console.error('Request error:', err);
        logEvent('error', 'detect-ringset-request-error', '', String(err));
        respond(500, JSON.stringify({ error: String(err) }));
      });
      req.on('end', async () => {
        try {
          const { filename, cx, cy } = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
          const imgPath = path.join(IMAGES_DIR, filename);
          const { rgba, width, height } = await loadImageNode(imgPath);
          const t = findRingSetFromCenter(rgba, width, height, cx, cy);
          t.paperBoundary.points = clampBoundary(t.paperBoundary.points, width, height) ?? [];
          respond(200, JSON.stringify({ rings: t.rings, paperBoundary: t.paperBoundary }));
        } catch (e) {
          console.error('detect-ringset error:', e);
          respond(500, JSON.stringify({ error: String(e) }));
        }
      });

    } else if (req.method === 'DELETE' && req.url?.startsWith('/api/image/')) {
      const filename = decodeURIComponent(req.url.slice('/api/image/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      try {
        await db.query('DELETE FROM annotations WHERE filename = $1', [filename]);
        await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
        const imgPath = path.join(IMAGES_DIR, filename);
        if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
        imageCache.delete(filename);
        inGenerated.delete(filename);
        inAnnotations.delete(filename);
        generationStatus.delete(filename);
        filenames.splice(filenames.indexOf(filename), 1);
        logEvent('info', 'delete', filename, 'image and all data removed');
        broadcastSSE({ type: 'removed', filename });
        respond(200, '{"ok":true}');
      } catch (err) {
        console.error('Delete error:', err);
        respond(500, JSON.stringify({ error: String(err) }));
      }

    } else {
      respond(404, '');
    }
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Annotation tool: http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop.');
    if (!process.env.NO_BROWSER) require('child_process').exec(`open http://localhost:${PORT}`);

    fs.watch(IMAGES_DIR, (event, name) => {
      if (!name || !/\.(jpg|jpeg)$/i.test(name)) return;
      if (filenames.includes(name)) return;
      const fullPath = path.join(IMAGES_DIR, name);
      setTimeout(() => {
        if (!fs.existsSync(fullPath)) return;
        console.log(`New image detected: ${name}`);
        filenames.push(name);
        generationStatus.set(name, 'queued');
        broadcastSSE({ type: 'new_image', filename: name });
      }, 500);
    });
  });
}

main().catch(err => { console.error(err); process.exit(1); });
