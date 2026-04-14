import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
import { Pool } from 'pg';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, findRingSetFromCenter, ArcheryResult } from '../src/targetDetection';
import { SplineRing } from '../src/spline';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const PORT = parseInt(process.env.ANNOTATE_PORT || '3737', 10);

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});
const K_POINTS = 8;

const RING_COLORS = [
  '#FFD700', '#FFD700',
  '#E8000D', '#E8000D',
  '#006CB7', '#006CB7',
  '#888888', '#888888',
  '#FFFFFF', '#FFFFFF',
];

interface TargetData {
  paperBoundary: [number, number][];
  ringSets: SplineRing[][];
}

interface CachedImageData {
  base64: string;
  width: number;
  height: number;
  detected: {
    targets: TargetData[];
    arrows: { tip: [number, number]; nock: [number, number] | null }[];
  };
}

interface ImageEntry {
  filename: string;
  base64: string;
  width: number;
  height: number;
  result: ArcheryResult;
}

// ---------------------------------------------------------------------------
// Logging (AW-6)
// ---------------------------------------------------------------------------

const LOG_PATH = path.resolve(__dirname, '../logs/annotate.log');
fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });

function logEvent(level: 'info' | 'warn' | 'error', event: string, filename: string, detail = '') {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, event, filename, detail });
  fs.appendFileSync(LOG_PATH, line + '\n');
  if (level === 'warn')  console.warn(`[warn]  ${event}: ${filename}${detail ? ' — ' + detail : ''}`);
  if (level === 'error') console.error(`[error] ${event}: ${filename}${detail ? ' — ' + detail : ''}`);
}

// ---------------------------------------------------------------------------
// Validity check (AW-2)
// ---------------------------------------------------------------------------

/** True if a boundary polygon is non-degenerate (non-empty, non-null, not all-zero). */
export function isBoundaryValid(pts: [number, number][]): boolean {
  if (pts.length === 0) return true;                             // absent = not yet set, OK
  return pts.length >= 3 &&
    pts.every(p => p[0] != null && p[1] != null) &&
    pts.some(p => p[0] !== 0 || p[1] !== 0);                   // at least one non-zero vertex
}

function isValidDetected(targets: TargetData[]): boolean {
  return targets.every(t =>
    isBoundaryValid(t.paperBoundary) &&
    Array.isArray(t.ringSets) &&
    t.ringSets.every(rs => rs.every(r => r.points?.every(p => p[0] != null && p[1] != null))),
  );
}

/**
 * True if a saved annotation has enough content to be worth keeping.
 *
 * Invalid cases (returns false):
 *   - no targets at all (empty annotation)
 *   - no arrows AND every target has a ringSets that is NOT a proper array
 *     (this catches old-format rows where `dbToTargets` produces `ringSets = SplineRing`
 *     instead of `SplineRing[][]`, which means the data was stored in the pre-multi-target
 *     flat format and was never properly annotated)
 */
function isValidAnnotation(targets: TargetData[], arrows: unknown[]): boolean {
  if (targets.length === 0) return false;
  if (arrows.length > 0) return true;
  // Without arrows, at least one target must have a properly-structured ringSets array.
  return targets.some(t => Array.isArray(t.ringSets));
}

// ---------------------------------------------------------------------------
// Worker process for background detection (AW-5)
// ---------------------------------------------------------------------------

const TSX_BIN = path.resolve(__dirname, '../node_modules/.bin/tsx');
const DETECT_WORKER = path.resolve(__dirname, 'detect-worker.ts');
const WORKER_TIMEOUT_MS = 5 * 60 * 1000;

interface DetectionOutput {
  rings: SplineRing[];
  paperBoundary: [number, number][] | null;
  arrows: { tip: [number, number]; nock: [number, number] | null }[];
  width?: number;
  height?: number;
  success: boolean;
  error?: string;
}

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

/** Convert DB format (paper_boundary: [x,y][][], rings: SplineRing[][][]) → TargetData[] */
function dbToTargets(rawBoundary: any, rawRings: any): TargetData[] {
  let raw = Array.isArray(rawBoundary) ? rawBoundary : [];
  // Migrate old single-polygon format [[x,y],...] → [[[x,y],...]] (one target).
  if (raw.length > 0 && !Array.isArray(raw[0]?.[0])) raw = [raw];
  const boundaries: [number, number][][] = raw;
  const ringsPerTarget: SplineRing[][][] = Array.isArray(rawRings) ? rawRings : [];
  const len = Math.max(boundaries.length, ringsPerTarget.length);
  if (len === 0) return [];
  return Array.from({ length: len }, (_, t) => ({
    paperBoundary: boundaries[t] ?? [],
    ringSets: ringsPerTarget[t] ?? [],
  }));
}

/** Convert TargetData[] → DB format { boundary: [x,y][][], rings: SplineRing[][][] } */
function targetsToDB(targets: any[]): { boundary: [number, number][][]; rings: SplineRing[][][] } {
  return {
    boundary: targets.map((t: any) => t.paperBoundary ?? []),
    rings:    targets.map((t: any) => t.ringSets ?? []),
  };
}

/** Wrap a single-target flat result → multi-target DB format */
function wrapSingleTarget(
  rings: SplineRing[],
  boundary: [number, number][] | null,
): { boundary: [number, number][][]; rings: SplineRing[][][] } {
  // Reject degenerate boundaries (null, too-short, or all-zero points).
  if (boundary && isBoundaryValid(boundary) && boundary.length >= 3) {
    return { boundary: [boundary], rings: [[rings]] };
  }
  return { boundary: [], rings: [] };
}

function runWorkerProcess(imgPath: string): Promise<DetectionOutput> {
  return new Promise((resolve, reject) => {
    const proc = spawn(TSX_BIN, [DETECT_WORKER, imgPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });
    const timer = setTimeout(() => {
      try { proc.kill('SIGTERM'); } catch {}
      reject(new Error('Worker timed out after 5 min'));
    }, WORKER_TIMEOUT_MS);
    proc.on('error', err => { clearTimeout(timer); reject(err); });
    proc.on('close', code => {
      clearTimeout(timer);
      if (code !== 0 && code !== null) {
        reject(new Error(`Worker exited ${code}: ${stderr.slice(0, 500)}`));
        return;
      }
      try { resolve(JSON.parse(stdout)); }
      catch { reject(new Error(`Failed to parse worker output: ${stdout.slice(0, 200)}`)); }
    });
  });
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

interface ProcessedImage extends ImageEntry {
  detectedArrows: { tip: [number, number]; nock: [number, number] | null }[];
}

async function processImage(imgPath: string): Promise<ProcessedImage> {
  const filename = path.basename(imgPath);
  console.log(`  [1/3] loadImageNode ${filename}…`);
  const { rgba, width, height } = await loadImageNode(imgPath);
  console.log(`  [2/3] loadImageBase64 ${filename} (${width}×${height})…`);
  const { base64 } = await loadImageBase64(imgPath);
  console.log(`  [3/3] findTarget ${filename}…`);
  const result = findTarget(rgba, width, height);
  console.log(`  done: ${filename}`);
  return { filename, base64, width, height, result, detectedArrows: [] };
}


function generateHtml(filenames: string[]): string {
  const filenamesJson = JSON.stringify(filenames);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ArcheryCounter — Annotation Tool</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #1a1a1a; color: #e0e0e0; font-family: system-ui, sans-serif; display: flex; height: 100vh; overflow: hidden; }

    #sidebar {
      width: 280px; min-width: 220px; max-width: 340px;
      background: #232323; border-right: 1px solid #333;
      display: flex; flex-direction: column; overflow: hidden;
    }
    #sidebar-header { padding: 12px 14px 8px; border-bottom: 1px solid #333; }
    #sidebar-header h1 { font-size: 1rem; color: #ccc; margin-bottom: 8px; }
    #controls { display: flex; flex-direction: column; gap: 6px; }
    #controls button { background: #2e6b3e; color: #fff; border: none; border-radius: 4px; padding: 6px 10px; cursor: pointer; font-size: 0.8rem; text-align: left; }
    #controls button:hover { background: #3a8a50; }
    #controls button.danger { background: #7a2222; }
    #controls button.danger:hover { background: #9a3232; }
    #controls button.active-mode { background: #8a5a00; }
    #controls button.active-mode:hover { background: #aa7200; }
    #controls button.save-disabled { background: #1e4428; color: #666; cursor: default; }
    #controls button.save-disabled:hover { background: #1e4428; }
    #controls button.secondary { background: #2a3a5a; }
    #controls button.secondary:hover { background: #3a4a7a; }
    #save-msg { font-size: 0.75rem; color: #e07070; padding: 4px 0 0; display: none; line-height: 1.4; }
    #controls .divider { border: none; border-top: 1px solid #333; margin: 2px 0; }

    #toolbar { padding: 8px 14px; border-bottom: 1px solid #333; display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
    #toolbar label { font-size: 0.78rem; color: #aaa; display: flex; align-items: center; gap: 4px; cursor: pointer; }
    #legend { font-size: 0.7rem; color: #666; width: 100%; margin-top: 2px; }

    #view-toggle { display: flex; gap: 0; border: 1px solid #444; border-radius: 4px; overflow: hidden; width: 100%; margin-bottom: 2px; }
    #view-toggle button {
      flex: 1; padding: 4px 8px; font-size: 0.78rem; border: none; cursor: pointer;
      background: #2a2a2a; color: #888; transition: background 0.15s, color 0.15s;
    }
    #view-toggle button:not(:last-child) { border-right: 1px solid #444; }
    #view-toggle button.active { background: #1a3a5c; color: #4a9eff; font-weight: 600; }
    #view-toggle button:hover:not(.active) { background: #333; color: #ccc; }
    #view-generated-badge {
      display: none; font-size: 0.68rem; color: #4a9eff; padding: 2px 6px;
      background: #0d2035; border-radius: 3px; width: 100%; text-align: center;
    }

    #score-picker {
      display: none; position: fixed; z-index: 200;
      gap: 4px; align-items: center; flex-wrap: nowrap;
      background: #1c1c1c; border: 1px solid #555; border-radius: 6px;
      padding: 7px 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.6);
      pointer-events: auto;
    }
    #score-picker .sp-label { font-size: 0.78rem; color: #FF8C00; white-space: nowrap; margin-right: 2px; }
    #score-picker button { padding: 3px 7px; font-size: 0.78rem; border: 1px solid #555; border-radius: 3px; background: #2a2a2a; color: #ddd; cursor: pointer; white-space: nowrap; }
    #score-picker button:hover { background: #3a3a3a; }
    #score-picker button.gold { background: #5a4a00; color: #FFD700; border-color: #FFD700; font-weight: bold; }
    #score-picker button.gold:hover { background: #7a6400; }
    #score-picker button.miss { background: #3a2a2a; color: #999; }
    #score-picker button.miss:hover { background: #4a3a3a; }

    #img-filter { display: flex; gap: 0; border: 1px solid #333; border-radius: 4px; overflow: hidden; margin: 6px 14px 0; flex-shrink: 0; }
    #img-filter button { flex: 1; padding: 4px 6px; font-size: 0.72rem; border: none; cursor: pointer; background: #2a2a2a; color: #666; }
    #img-filter button:not(:last-child) { border-right: 1px solid #333; }
    #img-filter button.active { background: #1a3a5c; color: #4a9eff; font-weight: 600; }
    #img-filter button:hover:not(.active) { background: #333; color: #aaa; }
    #img-list { flex: 1; overflow-y: auto; padding: 8px 0; }
    .img-btn {
      width: 100%; text-align: left; background: none; border: none; color: #ccc;
      padding: 7px 14px; font-size: 0.78rem; cursor: pointer; display: flex; align-items: center; gap: 6px;
      border-left: 3px solid transparent;
    }
    .img-btn:hover { background: #2a2a2a; }
    .img-btn.active { background: #1a3a5c; border-left-color: #4a9eff; color: #fff; }
    .img-btn.modified { color: #f0a030; }
    .img-btn .mod-dot { width: 6px; height: 6px; border-radius: 50%; background: #f0a030; flex-shrink: 0; }
    .img-btn .spacer  { width: 6px; flex-shrink: 0; }
    /* AW-5: generation status dot */
    .gen-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-left: auto; }
    .gen-dot.ready    { background: #4a9eff; }
    .gen-dot.annotated{ background: #2ecc71; }
    .gen-dot.computing{ background: #f0a030; animation: gen-pulse 1s ease-in-out infinite; }
    .gen-dot.queued   { background: #555; }
    .gen-dot.error    { background: #e74c3c; }
    @keyframes gen-pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

    /* AW-3: collapsible data panel */
    #data-panel { border-top: 1px solid #333; font-size: 0.72rem; flex-shrink: 0; }
    #data-panel-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 14px 6px; cursor: pointer; user-select: none;
    }
    #data-panel-header h3 { color: #888; font-size: 0.75rem; font-weight: 600; margin: 0; }
    #btn-collapse-data { background: none; border: none; color: #555; cursor: pointer; font-size: 0.75rem; padding: 0 2px; }
    #btn-collapse-data:hover { color: #aaa; }
    #data-table-wrap { padding: 0 14px 10px; overflow-y: auto; max-height: 220px; }
    #data-panel.collapsed #data-table-wrap { display: none; }
    #data-panel table { width: 100%; border-collapse: collapse; margin-bottom: 6px; }
    #data-panel th { color: #666; font-weight: 600; padding: 2px 4px; text-align: left; }
    #data-panel td { color: #aaa; font-family: monospace; padding: 2px 4px; border-top: 1px solid #2a2a2a; }
    #data-panel td.score-cell { cursor: pointer; color: #FF8C00; }
    #data-panel td.score-cell:hover { text-decoration: underline; }
    #data-table .target-header:hover { color: #4a9eff !important; }
    #data-table .rs-row:hover td { color: #ccc !important; cursor: pointer; }

    /* AW-4: suggested score highlight */
    #score-picker button.suggested { outline: 2px solid #4a9eff; outline-offset: 1px; }

    #main { flex: 1; display: flex; align-items: flex-start; justify-content: center; overflow: auto; background: #111; padding: 16px; position: relative; }
    #svg-container { position: relative; margin: auto; }
    .img-wrap { position: relative; display: inline-block; max-width: 100%; line-height: 0; }
    .img-wrap img { display: block; max-width: 100%; height: auto; user-select: none; -webkit-user-drag: none; }
    .img-wrap svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: visible; cursor: default; }
    .img-wrap svg.dragging { cursor: grabbing; }
    .img-wrap svg.add-arrow-mode { cursor: crosshair; }
    .handle { cursor: grab; }
    .handle:active { cursor: grabbing; }

    .loading-wrap {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 12px; color: #888; font-size: 0.85rem; min-width: 300px; min-height: 200px;
    }
    .loading-bar-track {
      width: 260px; height: 4px; background: #2a2a2a; border-radius: 2px; overflow: hidden;
    }
    .loading-bar-fill {
      height: 100%; width: 40%; background: #4a9eff; border-radius: 2px;
      animation: loading-slide 1.2s ease-in-out infinite;
    }
    @keyframes loading-slide {
      0%   { transform: translateX(-100%); }
      100% { transform: translateX(750%); }
    }
  </style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Annotation Tool</h1>
    <div id="controls">
      <button id="btn-save">Save</button>
      <div id="save-msg"></div>
      <button id="btn-add-arrow">Add arrow (A)</button>
      <button id="btn-add-ringset">Add ring set (R)</button>
      <button id="btn-remove-ringset" class="secondary">Remove ring set</button>
      <hr class="divider"/>
      <button id="btn-reset-boundaries" class="secondary">Reset boundaries</button>
      <button id="btn-reset-rings" class="secondary">Reset rings</button>
      <button id="btn-reset-arrows" class="secondary">Reset arrows</button>
      <button id="btn-reset">Reset all</button>
      <button id="btn-delete" class="danger">Delete image</button>
    </div>
  </div>
  <div id="toolbar">
    <div id="view-toggle">
      <button id="btn-view-annotated" class="active">Annotated</button>
      <button id="btn-view-generated">Generated</button>
    </div>
    <div id="view-generated-badge">read-only · algorithm output</div>
    <label><input type="checkbox" id="chk-rings" checked/> Rings</label>
    <label><input type="checkbox" id="chk-boundary" checked/> Boundary</label>
    <label><input type="checkbox" id="chk-arrows" checked/> Arrows</label>
    <div id="legend">
      Ctrl+click boundary: add vertex · Alt+click ring: add point · Shift+click vertex/ring pt/arrow: remove · A: add arrow · R: add ring set
    </div>
  </div>
  <div id="img-filter">
    <button id="filter-all" class="active">All</button>
    <button id="filter-annotated">Annotated</button>
    <button id="filter-unannotated">Unannotated</button>
  </div>
  <div id="img-list"></div>
  <div id="data-panel">
    <div id="data-panel-header">
      <h3>Current annotation</h3>
      <button id="btn-collapse-data" title="Toggle panel">▼</button>
    </div>
    <div id="data-table-wrap"><div id="data-table"></div></div>
  </div>
</div>

<div id="main">
  <div id="svg-container"></div>
  <div id="score-picker">
    <span class="sp-label" id="score-prompt">Score:</span>
    <button class="gold" data-score="X">X</button>
    <button class="gold" data-score="10">10</button>
    <button data-score="9">9</button>
    <button data-score="8">8</button>
    <button data-score="7">7</button>
    <button data-score="6">6</button>
    <button data-score="5">5</button>
    <button data-score="4">4</button>
    <button data-score="3">3</button>
    <button data-score="2">2</button>
    <button data-score="1">1</button>
    <button class="miss" data-score="0">M</button>
    <button class="miss" id="score-skip">?</button>
  </div>
</div>

<script>
const IMAGES = ${filenamesJson};
const RING_COLORS = ${JSON.stringify(RING_COLORS)};
const K_POINTS = ${K_POINTS};

// ---- Catmull-Rom spline (browser-side) ----
function evalCatmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return [
    0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
    0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3),
  ];
}

function sampleClosedSpline(pts, nSamples) {
  const K = pts.length;
  if (K < 2) return pts.map(p => [...p]);
  const out = [];
  const sps = Math.ceil(nSamples / K);
  for (let k = 0; k < K; k++) {
    const p0 = pts[(k-1+K)%K], p1 = pts[k], p2 = pts[(k+1)%K], p3 = pts[(k+2)%K];
    for (let s = 0; s < sps; s++) out.push(evalCatmullRom(p0, p1, p2, p3, s/sps));
  }
  return out;
}

function splineToPath(pts) {
  const sampled = sampleClosedSpline(pts, 120);
  return sampled.map((p, i) => \`\${i===0?'M':'L'}\${p[0].toFixed(1)},\${p[1].toFixed(1)}\`).join(' ') + ' Z';
}

// ---- Geometry helpers (AW-4) ----
function ptInPoly(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

function deduceScore(tipX, tipY, rings, boundary) {
  for (let i = 0; i < rings.length; i++) {
    if (!rings[i] || !rings[i].points || rings[i].points.length < 3) continue;
    if (rings[i].points.some(p => p[0] == null)) continue;
    const poly = sampleClosedSpline(rings[i].points, 64);
    if (ptInPoly(tipX, tipY, poly)) return 10 - i;
  }
  if (boundary && boundary.length >= 3 && ptInPoly(tipX, tipY, boundary)) return 1;
  return 0;
}

// ---- State ----
let store = { annotations: {}, modified: [] };
let staleImages = new Set(); // filenames with stale/missing detections
let generationStatus = {}; // filename → 'ready'|'queued'|'computing'
let currentIdx = 0;
let currentTargetIdx = 0;
let currentRingSetIdx = 0;
let drag = null;
let addArrowMode = 'idle';   // 'idle' | 'place-tip' | 'score-input'
let addRingSetMode = 'idle'; // 'idle' | 'place-center'
let pendingTip = null;
let pickerContext = null; // { type: 'new' } | { type: 'edit', ai: number } | null
let imageDataCache = {}; // filename -> { base64, width, height, detected }
let viewMode = 'annotated'; // 'annotated' | 'generated'
let imageFilter = 'all'; // 'all' | 'annotated' | 'unannotated'

// ---- Overlay toggles ----
function showRings()    { return document.getElementById('chk-rings').checked; }
function showBoundary() { return document.getElementById('chk-boundary').checked; }
function showArrows()   { return document.getElementById('chk-arrows').checked; }

// ---- Annotation helpers ----
function getDetected(idx) {
  const data = imageDataCache[IMAGES[idx]];
  if (!data) return { targets: [] };
  return { targets: data.detected.targets || [] };
}

function getAnnotation(idx) {
  const filename = IMAGES[idx];
  if (!store.annotations[filename]) {
    const detected = getDetected(idx);
    store.annotations[filename] = {
      targets: detected.targets.map(t => ({
        paperBoundary: t.paperBoundary.map(p => [p[0], p[1]]),
        ringSets: t.ringSets.map(rs => rs.map(r => ({ points: r.points.map(p => [p[0], p[1]]) }))),
      })),
      arrows: [],
    };
  }
  const ann = store.annotations[filename];
  if (!ann.arrows)  ann.arrows  = [];
  if (!ann.targets) ann.targets = [];
  return ann;
}

function isAnnotationValid(_ann) {
  return true;
}

function markModified(idx) {
  const filename = IMAGES[idx];
  if (!store.modified.includes(filename)) store.modified.push(filename);
}

// ---- SVG coordinate helper ----
function svgPt(svg, e) {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

// ---- Boundary edge insertion ----
function nearestBoundaryEdge(boundary, x, y) {
  let bestEdge = -1, bestDist = Infinity, bestPt = null;
  const n = boundary.length;
  for (let i = 0; i < n; i++) {
    const a = boundary[i], b = boundary[(i+1)%n];
    const dx = b[0]-a[0], dy = b[1]-a[1];
    const len2 = dx*dx + dy*dy;
    if (len2 < 1) continue;
    const t = Math.max(0, Math.min(1, ((x-a[0])*dx + (y-a[1])*dy) / len2));
    const px = a[0]+t*dx, py = a[1]+t*dy;
    const dist = Math.hypot(x-px, y-py);
    if (dist < bestDist) { bestDist = dist; bestEdge = i; bestPt = [px, py]; }
  }
  return { edge: bestEdge, dist: bestDist, pt: bestPt };
}

// ---- Ring segment insertion ----
function nearestRingSegment(rings, x, y) {
  let bestRi = -1, bestSeg = -1, bestDist = Infinity;
  rings.forEach((ring, ri) => {
    if (!ring.points || ring.points.length < 3) return;
    const K = ring.points.length;
    const sps = Math.ceil(120 / K);
    const samples = sampleClosedSpline(ring.points, 120);
    samples.forEach((s, k) => {
      const nx = samples[(k + 1) % samples.length];
      const dx = nx[0] - s[0], dy = nx[1] - s[1];
      const len2 = dx * dx + dy * dy;
      let dist;
      if (len2 < 1) {
        dist = Math.hypot(x - s[0], y - s[1]);
      } else {
        const t = Math.max(0, Math.min(1, ((x - s[0]) * dx + (y - s[1]) * dy) / len2));
        dist = Math.hypot(x - (s[0] + t * dx), y - (s[1] + t * dy));
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestRi = ri;
        bestSeg = Math.floor(k / sps) % K;
      }
    });
  });
  return { ri: bestRi, seg: bestSeg, dist: bestDist };
}

// ---- Score picker (AW-4: suggested score) ----
function showScorePicker(label, suggested, clientX, clientY) {
  const picker = document.getElementById('score-picker');
  document.getElementById('score-prompt').textContent = label || 'Score:';
  picker.querySelectorAll('button[data-score]').forEach(btn => {
    const raw = btn.getAttribute('data-score');
    const val = raw === 'X' ? 'X' : parseInt(raw, 10);
    btn.classList.toggle('suggested', suggested !== undefined && val === suggested);
  });
  picker.style.display = 'flex';
  if (clientX !== undefined && clientY !== undefined) {
    const pw = picker.offsetWidth, ph = picker.offsetHeight;
    const px = Math.max(4, Math.min(clientX - pw / 2, window.innerWidth - pw - 4));
    const idealPy = clientY - ph - 8;
    const py = Math.max(4, Math.min(window.innerHeight - ph - 4,
               Math.min(clientY + 50, Math.max(clientY - ph - 50, idealPy))));
    picker.style.left = px + 'px';
    picker.style.top  = py + 'px';
  }
}

function hideScorePicker() {
  document.getElementById('score-picker').style.display = 'none';
  pickerContext = null;
}

function onScoreSelect(score) {
  if (!pickerContext) return;
  const ann = getAnnotation(currentIdx);
  if (pickerContext.type === 'new') {
    ann.arrows.push({
      tip:  [Math.round(pendingTip[0]),  Math.round(pendingTip[1])],
      nock: null,
      score,
    });
    pendingTip = null;
    addArrowMode = 'idle';
    document.getElementById('btn-add-arrow').classList.remove('active-mode');
    hideScorePicker();
    markModified(currentIdx);
    updateImageList();
    render();
  } else if (pickerContext.type === 'edit') {
    ann.arrows[pickerContext.ai].score = score;
    hideScorePicker();
    markModified(currentIdx);
    updateImageList();
    render();
    updateDataPanel();
  }
}

// ---- View mode toggle ----
function setViewMode(mode) {
  viewMode = mode;
  document.getElementById('btn-view-annotated').classList.toggle('active', mode === 'annotated');
  document.getElementById('btn-view-generated').classList.toggle('active', mode === 'generated');
  document.getElementById('view-generated-badge').style.display = mode === 'generated' ? 'block' : 'none';
  if (mode === 'generated' && addArrowMode !== 'idle') {
    pendingTip = null;
    addArrowMode = 'idle';
    hideScorePicker();
    document.getElementById('btn-add-arrow').classList.remove('active-mode');
  }
  if (mode === 'generated' && addRingSetMode !== 'idle') {
    addRingSetMode = 'idle';
    document.getElementById('btn-add-ringset').classList.remove('active-mode');
  }
  render();
}

// ---- Loading indicator ----
function renderLoading(message) {
  const container = document.getElementById('svg-container');
  container.innerHTML = \`<div class="loading-wrap">
    <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
    <div id="loading-msg">\${message || 'Loading\u2026'}</div>
  </div>\`;
}

function updateLoadingMessage(message) {
  const el = document.getElementById('loading-msg');
  if (el) el.textContent = message;
}

// ---- Select image (with lazy load) ----
async function selectImage(idx) {
  currentIdx = idx;
  currentTargetIdx = 0;
  currentRingSetIdx = 0;
  updateImageList();
  const filename = IMAGES[idx];
  console.log('[selectImage] clicked:', filename, 'cached:', !!imageDataCache[filename], 'stale:', staleImages.has(filename));
  if (imageDataCache[filename]) {
    render();
    return;
  }

  let computing = staleImages.has(filename);
  renderLoading(computing ? 'Computing rings\u2026 0s' : 'Loading\u2026');

  try {
    console.log('[selectImage] fetching status…');
    const r = await fetch('/api/image-status/' + encodeURIComponent(filename));
    const { state } = await r.json();
    computing = state === 'stale' || state === 'new';
    console.log('[selectImage] status:', state, 'computing:', computing);
    updateLoadingMessage(computing ? 'Computing rings\u2026 0s' : 'Loading\u2026');
  } catch (e) { console.warn('[selectImage] status fetch failed:', e); }

  let elapsed = 0;
  const timer = computing ? setInterval(() => {
    elapsed++;
    updateLoadingMessage(\`Computing rings\u2026 \${elapsed}s\`);
  }, 1000) : null;

  let loadError = null;
  try {
    console.log('[selectImage] fetching image data…');
    const t0 = Date.now();
    const res = await fetch('/api/image/' + encodeURIComponent(filename));
    console.log('[selectImage] response received in', ((Date.now()-t0)/1000).toFixed(1)+'s  status:', res.status);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || 'HTTP ' + res.status);
    }
    console.log('[selectImage] parsing JSON…');
    imageDataCache[filename] = await res.json();
    const det = imageDataCache[filename]?.detected;
    console.log('[selectImage] done, targets:', det?.targets?.length, 'arrows:', det?.arrows?.length);
    staleImages.delete(filename);
    if (!store.annotations[filename]) {
      try {
        const annRes = await fetch('/api/annotation/' + encodeURIComponent(filename));
        if (annRes.ok) {
          const ann = await annRes.json();
          if (ann) store.annotations[filename] = ann;
        }
      } catch (e) { console.warn('[selectImage] annotation reload failed:', e); }
    }
  } catch (e) {
    console.error('[selectImage] failed:', filename, e);
    loadError = String(e);
  } finally {
    if (timer) clearInterval(timer);
  }
  if (loadError) {
    renderLoading('\u26a0 ' + loadError);
    return;
  }
  render();
}

// ---- Render ----
function render() {
  const container = document.getElementById('svg-container');
  const filename = IMAGES[currentIdx];
  const data = imageDataCache[filename];
  if (!data) return;
  const ann = getAnnotation(currentIdx);

  const isGenerated = viewMode === 'generated';
  const targets = isGenerated ? (data.detected.targets || []) : ann.targets;
  const arrows  = isGenerated ? (data.detected.arrows  || []) : ann.arrows;

  const W = data.width, H = data.height;
  const showR = showRings(), showB = showBoundary(), showA = showArrows();

  let svgContent = '';

  targets.forEach((target, ti) => {
    const isActiveTarget = ti === currentTargetIdx;
    const boundary = target.paperBoundary || [];
    const ringSets = target.ringSets || [];

    // Boundary polygon
    if (showB && boundary.length >= 3 && boundary.every(p => p[0] != null && p[1] != null)) {
      const opacity = isActiveTarget ? 0.85 : 0.3;
      const pts = boundary.map(p => \`\${p[0].toFixed(1)},\${p[1].toFixed(1)}\`).join(' ');
      svgContent += \`<polygon points="\${pts}" fill="none" stroke="#00FF88" stroke-width="3" stroke-dasharray="12 6" opacity="\${opacity}"/>\`;
    }

    // Boundary vertex handles (active target, annotated mode only)
    if (showB && isActiveTarget && !isGenerated && boundary.length > 0) {
      boundary.forEach((p, i) => {
        svgContent += \`<circle class="handle" data-handle="boundary" data-ti="\${ti}" data-idx="\${i}" cx="\${p[0].toFixed(1)}" cy="\${p[1].toFixed(1)}" r="10" fill="#00FF88" stroke="#000" stroke-width="2" opacity="0.9"/>\`;
        svgContent += \`<text x="\${p[0].toFixed(1)}" y="\${(p[1]-14).toFixed(1)}" text-anchor="middle" fill="#00FF88" font-size="11" font-family="monospace" pointer-events="none">\${i}</text>\`;
      });
    }

    // Ring sets
    ringSets.forEach((rings, si) => {
      const isActiveRS = isActiveTarget && si === currentRingSetIdx;
      const ringOpacity = isActiveRS ? 1.0 : 0.3;

      if (showR && rings.length > 0) {
        // Draw outermost first
        for (let i = rings.length - 1; i >= 0; i--) {
          const ring = rings[i];
          if (!ring.points || ring.points.length < 3) continue;
          if (ring.points.some(p => p[0] == null || p[1] == null)) continue;
          const color = RING_COLORS[i] || '#FFFFFF';
          const d = splineToPath(ring.points);
          if (i >= 8) {
            svgContent += \`<path d="\${d}" fill="none" stroke="#000" stroke-width="4" opacity="\${ringOpacity}"/>\`;
            svgContent += \`<path d="\${d}" fill="none" stroke="\${color}" stroke-width="2" opacity="\${ringOpacity}"/>\`;
          } else if (i >= 6) {
            svgContent += \`<path d="\${d}" fill="none" stroke="#888" stroke-width="4" opacity="\${ringOpacity}"/>\`;
            svgContent += \`<path d="\${d}" fill="none" stroke="\${color}" stroke-width="2" opacity="\${ringOpacity}"/>\`;
          } else {
            svgContent += \`<path d="\${d}" fill="none" stroke="\${color}" stroke-width="2" opacity="\${ringOpacity}"/>\`;
          }
        }

        // Badge T{ti+1}.{si+1} on outermost ring
        const outerRing = rings[rings.length - 1];
        if (outerRing?.points?.length > 0) {
          const bcx = (outerRing.points.reduce((s, p) => s + p[0], 0) / outerRing.points.length).toFixed(1);
          const bcy = (outerRing.points.reduce((s, p) => s + p[1], 0) / outerRing.points.length).toFixed(1);
          const badgeLabel = \`T\${ti+1}.\${si+1}\`;
          const badgeColor = isActiveRS ? '#4a9eff' : '#888';
          svgContent += \`<text x="\${bcx}" y="\${bcy}" text-anchor="middle" dominant-baseline="middle" fill="\${badgeColor}" font-size="14" font-weight="bold" font-family="monospace" pointer-events="none" opacity="\${isActiveRS ? 1 : 0.5}">\${badgeLabel}</text>\`;
        }

        // Control point handles (active ring set, annotated mode only)
        if (isActiveRS && !isGenerated) {
          rings.forEach((ring, ri) => {
            if (!ring.points) return;
            const color = RING_COLORS[ri] || '#FFFFFF';
            const labelFill = (ri === 6 || ri === 7) ? '#fff' : (ri >= 8 ? '#222' : '#111');
            ring.points.forEach((p, pi) => {
              svgContent += \`<circle class="handle" data-handle="ring_pt" data-ti="\${ti}" data-si="\${si}" data-ri="\${ri}" data-pi="\${pi}" cx="\${p[0].toFixed(1)}" cy="\${p[1].toFixed(1)}" r="4" fill="\${color}" stroke="#000" stroke-width="1" opacity="0.85"/>\`;
              if (pi === 0) {
                svgContent += \`<text x="\${p[0].toFixed(1)}" y="\${(p[1]+3.5).toFixed(1)}" text-anchor="middle" dominant-baseline="middle" fill="\${labelFill}" font-size="8" font-weight="bold" font-family="monospace" pointer-events="none">\${ri}</text>\`;
              }
            });
          });
        }
      }
    });
  });

  // Detected arrows (generated mode only)
  if (showA && isGenerated && arrows) {
    arrows.forEach((arrow, ai) => {
      const { tip, nock } = arrow;
      svgContent += \`<line x1="\${tip[0].toFixed(1)}" y1="\${tip[1].toFixed(1)}" x2="\${(nock ? nock[0] : tip[0]).toFixed(1)}" y2="\${(nock ? nock[1] : tip[1]).toFixed(1)}" stroke="#00CFCF" stroke-width="2" stroke-dasharray="8 4"/>\`;
      svgContent += \`<circle cx="\${tip[0].toFixed(1)}" cy="\${tip[1].toFixed(1)}" r="5" fill="#00CFCF" stroke="#000" stroke-width="1"/>\`;
      if (nock) {
        svgContent += \`<circle cx="\${nock[0].toFixed(1)}" cy="\${nock[1].toFixed(1)}" r="4" fill="#00FFFF" stroke="#000" stroke-width="1" opacity="0.7"/>\`;
      }
      const mx = ((tip[0] + (nock ? nock[0] : tip[0])) / 2).toFixed(1);
      const my = ((tip[1] + (nock ? nock[1] : tip[1])) / 2 - 14).toFixed(1);
      svgContent += \`<text x="\${mx}" y="\${my}" text-anchor="middle" fill="#00CFCF" font-size="10" font-weight="bold" font-family="monospace" pointer-events="none">\${ai}</text>\`;
    });
  }

  // Annotated arrows
  if (showA && !isGenerated) {
    ann.arrows.forEach((arrow, ai) => {
      const { tip, score } = arrow;
      const isMiss = score === 0;
      const isIncomplete = score === null || score === undefined;
      const dotColor = isMiss ? '#888888' : isIncomplete ? '#aaaaaa' : '#FF4500';
      const labelText = score === 0 ? 'M' : isIncomplete ? '?' : String(score);
      const mx = tip[0].toFixed(1);
      const my = (tip[1] - 14).toFixed(1);
      const tipDash = isIncomplete ? 'stroke-dasharray="3 2"' : '';
      svgContent += \`<circle class="handle" data-handle="arrow_tip" data-ai="\${ai}" cx="\${tip[0].toFixed(1)}" cy="\${tip[1].toFixed(1)}" r="6" fill="\${dotColor}" stroke="#000" stroke-width="1.5" \${tipDash}/>\`;
      svgContent += \`<text x="\${mx}" y="\${my}" text-anchor="middle" fill="white" font-size="10" font-weight="bold" font-family="monospace" pointer-events="none">\${labelText}</text>\`;
    });

    if (addArrowMode === 'score-input' && pendingTip) {
      svgContent += \`<circle cx="\${pendingTip[0].toFixed(1)}" cy="\${pendingTip[1].toFixed(1)}" r="6" fill="#FF8C00" opacity="0.5" pointer-events="none"/>\`;
    }
  }

  const inCrosshairMode = addArrowMode === 'place-tip' || addRingSetMode === 'place-center';
  const svgClass = inCrosshairMode ? 'class="add-arrow-mode"' : '';
  const wrapEl = \`<div class="img-wrap">
  <img src="\${data.base64}" alt="\${filename}" draggable="false"/>
  <svg id="main-svg" viewBox="0 0 \${W} \${H}" \${svgClass} xmlns="http://www.w3.org/2000/svg">
    \${svgContent}
  </svg>
</div>\`;

  container.innerHTML = wrapEl;
  attachSvgListeners();
  updateDataPanel();
}

// ---- Add ring set (detect-ringset flow) ----
async function detectAndAddRingSet(filename, cx, cy) {
  // Capture idx before any await so we always modify the correct image's annotation
  const savedIdx = currentIdx;
  console.log(\`[detect-ringset] start — \${filename} idx=\${savedIdx} seed=(\${Math.round(cx)},\${Math.round(cy)})\`);
  showStatusMsg('Detecting ring set\u2026');
  try {
    const t0 = Date.now();
    const res = await fetch('/api/detect-ringset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, cx, cy }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const msg = body.error || 'HTTP ' + res.status;
      console.error(\`[detect-ringset] FAILED: \${msg}\`);
      showStatusMsg('\u26a0 Detection failed: ' + msg, 5000);
      return;
    }
    const { rings, paperBoundary } = await res.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    if (!rings || rings.length === 0) {
      console.warn(\`[detect-ringset] no rings returned (\${elapsed}s)\`);
      showStatusMsg('\u26a0 No rings detected at that location.', 5000);
      return;
    }
    console.log(\`[detect-ringset] got \${rings.length} rings, boundary \${paperBoundary?.length ?? 0} pts (\${elapsed}s)\`);

    // Use savedIdx (captured before await) to avoid stale-currentIdx race
    const ann = getAnnotation(savedIdx);
    console.log(\`[detect-ringset] annotation targets before: \${ann.targets.length}\`);

    // Find centroid of outermost ring to check target overlap
    const outerRing = rings[rings.length - 1];
    let ringCx = cx, ringCy = cy;
    if (outerRing?.points?.length > 0) {
      ringCx = outerRing.points.reduce((s, p) => s + p[0], 0) / outerRing.points.length;
      ringCy = outerRing.points.reduce((s, p) => s + p[1], 0) / outerRing.points.length;
    }
    console.log(\`[detect-ringset] outermost ring centroid: (\${Math.round(ringCx)},\${Math.round(ringCy)})\`);

    // Check if centroid falls inside any existing target boundary
    let targetIdx = -1;
    for (let t = 0; t < ann.targets.length; t++) {
      const pb = ann.targets[t].paperBoundary;
      console.log(\`[detect-ringset]   checking target \${t}: boundary \${pb.length} pts → \${pb.length >= 3 ? ptInPoly(ringCx, ringCy, pb) : 'too few pts'}\`);
      if (pb.length >= 3 && ptInPoly(ringCx, ringCy, pb)) { targetIdx = t; break; }
    }

    let statusMsg;
    if (targetIdx >= 0) {
      ann.targets[targetIdx].ringSets.push(rings);
      currentTargetIdx = targetIdx;
      currentRingSetIdx = ann.targets[targetIdx].ringSets.length - 1;
      statusMsg = \`Ring set RS\${currentRingSetIdx + 1} added to target T\${currentTargetIdx + 1}\`;
    } else {
      ann.targets.push({ paperBoundary: paperBoundary || [], ringSets: [rings] });
      currentTargetIdx = ann.targets.length - 1;
      currentRingSetIdx = 0;
      statusMsg = \`New target T\${currentTargetIdx + 1} created with \${paperBoundary?.length ?? 0}-pt boundary\`;
    }
    console.log(\`[detect-ringset] \${statusMsg} — targets after: \${ann.targets.length}, ringSets: \${ann.targets.map(t => t.ringSets.length).join(',')}\`);

    showStatusMsg(statusMsg, 4000);

    // If the user navigated to a different image during detection, switch back
    if (currentIdx !== savedIdx) {
      console.warn(\`[detect-ringset] currentIdx changed (\${savedIdx}→\${currentIdx}) during detection; result saved to annotation, not switching back\`);
    }
    markModified(savedIdx);
    updateImageList();
    render();
  } catch (e) {
    console.error('[detect-ringset] error:', e);
    showStatusMsg('\u26a0 Detection error: ' + e, 5000);
  }
}

function removeActiveRingSet() {
  const ann = getAnnotation(currentIdx);
  if (currentTargetIdx >= ann.targets.length) return;
  const target = ann.targets[currentTargetIdx];
  target.ringSets.splice(currentRingSetIdx, 1);
  if (target.ringSets.length === 0) {
    ann.targets.splice(currentTargetIdx, 1);
    currentTargetIdx = Math.max(0, currentTargetIdx - 1);
    currentRingSetIdx = 0;
  } else {
    currentRingSetIdx = Math.min(currentRingSetIdx, target.ringSets.length - 1);
  }
  markModified(currentIdx);
  updateImageList();
  render();
}

// ---- Drag & click handling ----
function attachSvgListeners() {
  const svg = document.getElementById('main-svg');
  if (!svg) return;

  svg.addEventListener('click', async (e) => {
    if (viewMode === 'generated') return;

    // Add ring set mode — place center
    if (addRingSetMode === 'place-center') {
      addRingSetMode = 'idle';
      document.getElementById('btn-add-ringset').classList.remove('active-mode');
      const mpt = svgPt(svg, e);
      render(); // remove crosshair cursor
      await detectAndAddRingSet(IMAGES[currentIdx], mpt.x, mpt.y);
      return;
    }

    if (e.ctrlKey || e.metaKey) {
      const ann = getAnnotation(currentIdx);
      const target = ann.targets[currentTargetIdx];
      if (!target || target.paperBoundary.length < 3) return;
      const mpt = svgPt(svg, e);
      const { edge, pt } = nearestBoundaryEdge(target.paperBoundary, mpt.x, mpt.y);
      if (edge >= 0) {
        const imgData = imageDataCache[IMAGES[currentIdx]];
        const w = imgData?.width ?? 0, h = imgData?.height ?? 0;
        target.paperBoundary.splice(edge + 1, 0, [
          Math.round(Math.max(0, Math.min(w - 1, pt[0]))),
          Math.round(Math.max(0, Math.min(h - 1, pt[1]))),
        ]);
        markModified(currentIdx);
        updateImageList();
        render();
      }
      return;
    }

    if (e.altKey) {
      const ann = getAnnotation(currentIdx);
      const target = ann.targets[currentTargetIdx];
      const rings = target?.ringSets[currentRingSetIdx];
      if (!rings || rings.length === 0) return;
      const mpt = svgPt(svg, e);
      const { ri, seg } = nearestRingSegment(rings, mpt.x, mpt.y);
      if (ri >= 0) {
        rings[ri].points.splice(seg + 1, 0, [mpt.x, mpt.y]);
        markModified(currentIdx);
        updateImageList();
        render();
      }
      return;
    }

    if (e.target.closest && e.target.closest('.handle')) return;

    if (addArrowMode === 'place-tip') {
      const mpt = svgPt(svg, e);
      pendingTip = [mpt.x, mpt.y];
      addArrowMode = 'score-input';
      pickerContext = { type: 'new' };
      const ann = getAnnotation(currentIdx);
      const target = ann.targets[currentTargetIdx];
      const rings  = target?.ringSets[currentRingSetIdx] ?? [];
      const boundary = target?.paperBoundary ?? null;
      const suggested = deduceScore(pendingTip[0], pendingTip[1], rings, boundary);
      showScorePicker('Score:', suggested === 0 ? 0 : suggested || undefined, e.clientX, e.clientY);
      render();
    }
  });

  svg.querySelectorAll('.handle').forEach(el => {
    el.addEventListener('click', (e) => {
      if (viewMode === 'generated') return;
      if (!e.shiftKey) return;
      e.preventDefault(); e.stopPropagation();
      const handleType = el.getAttribute('data-handle');
      const ann = getAnnotation(currentIdx);
      if (handleType === 'boundary') {
        const ti  = parseInt(el.getAttribute('data-ti'),  10);
        const idx = parseInt(el.getAttribute('data-idx'), 10);
        const target = ann.targets[ti];
        if (!target || target.paperBoundary.length <= 3) return;
        target.paperBoundary.splice(idx, 1);
        markModified(currentIdx);
        updateImageList();
        render();
      } else if (handleType === 'ring_pt') {
        const ti = parseInt(el.getAttribute('data-ti'), 10);
        const si = parseInt(el.getAttribute('data-si'), 10);
        const ri = parseInt(el.getAttribute('data-ri'), 10);
        const pi = parseInt(el.getAttribute('data-pi'), 10);
        const ring = ann.targets[ti]?.ringSets[si]?.[ri];
        if (ring && ring.points.length > 3) {
          ring.points.splice(pi, 1);
          markModified(currentIdx);
          updateImageList();
          render();
        }
      } else if (handleType === 'arrow_tip') {
        const ai = parseInt(el.getAttribute('data-ai'), 10);
        ann.arrows.splice(ai, 1);
        markModified(currentIdx);
        updateImageList();
        render();
      }
    });

    el.addEventListener('mousedown', (e) => {
      if (viewMode === 'generated') return;
      if (e.shiftKey || e.ctrlKey || e.metaKey) return;
      if (addArrowMode !== 'idle' || addRingSetMode !== 'idle') return;
      e.preventDefault(); e.stopPropagation();
      const handleType = el.getAttribute('data-handle');

      if (handleType === 'ring_pt') {
        const ti = parseInt(el.getAttribute('data-ti'), 10);
        const si = parseInt(el.getAttribute('data-si'), 10);
        if (ti !== currentTargetIdx || si !== currentRingSetIdx) {
          currentTargetIdx = ti; currentRingSetIdx = si; render();
        }
        drag = { type: 'ring_pt', ti, si, ri: parseInt(el.getAttribute('data-ri'), 10), pi: parseInt(el.getAttribute('data-pi'), 10) };
      } else if (handleType === 'boundary') {
        const ti = parseInt(el.getAttribute('data-ti'), 10);
        if (ti !== currentTargetIdx) { currentTargetIdx = ti; render(); }
        drag = { type: 'boundary', ti, idx: parseInt(el.getAttribute('data-idx'), 10) };
      } else if (handleType === 'arrow_tip') {
        drag = { type: 'arrow_tip', ai: parseInt(el.getAttribute('data-ai'), 10) };
      }

      const svgEl0 = document.getElementById('main-svg');
      if (svgEl0) svgEl0.classList.add('dragging');

      const onMove = (me) => {
        if (!drag) return;
        const liveSvg = document.getElementById('main-svg');
        if (!liveSvg) return;
        const mpt = svgPt(liveSvg, me);
        const ann = getAnnotation(currentIdx);

        if (drag.type === 'ring_pt') {
          const ring = ann.targets[drag.ti]?.ringSets[drag.si]?.[drag.ri];
          if (ring) ring.points[drag.pi] = [mpt.x, mpt.y];
        } else if (drag.type === 'boundary') {
          const target = ann.targets[drag.ti];
          if (target) {
            const imgData = imageDataCache[IMAGES[currentIdx]];
            const w = imgData?.width ?? 0, h = imgData?.height ?? 0;
            target.paperBoundary[drag.idx] = [
              Math.round(Math.max(0, Math.min(w - 1, mpt.x))),
              Math.round(Math.max(0, Math.min(h - 1, mpt.y))),
            ];
          }
        } else if (drag.type === 'arrow_tip') {
          ann.arrows[drag.ai].tip = [mpt.x, mpt.y];
        }
        render();
      };

      const onUp = () => {
        drag = null;
        const svgElUp = document.getElementById('main-svg');
        if (svgElUp) svgElUp.classList.remove('dragging');
        markModified(currentIdx);
        updateImageList();
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      };

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  });
}

// ---- Image list ----
function isAnnotated(filename) {
  const ann = store.annotations[filename];
  return ann != null && ann.arrows && ann.arrows.length > 0;
}

function updateImageList() {
  const list = document.getElementById('img-list');
  list.innerHTML = '';
  IMAGES.forEach((filename, i) => {
    const annotated = isAnnotated(filename);
    if (imageFilter === 'annotated'   && !annotated) return;
    if (imageFilter === 'unannotated' &&  annotated) return;

    const isModified = store.modified.includes(filename);
    const isActive   = i === currentIdx;
    const genState   = generationStatus[filename] || (staleImages.has(filename) ? 'queued' : 'ready');
    const btn = document.createElement('button');
    btn.className = 'img-btn' + (isActive ? ' active' : '') + (isModified ? ' modified' : '');
    const modDot = isModified ? '<span class="mod-dot"></span>' : '<span class="spacer"></span>';
    const dotClass = annotated ? 'annotated' : genState;
    const dotTitle = annotated ? 'has annotation' : genState === 'ready' ? 'detected' : genState === 'computing' ? 'computing…' : genState === 'error' ? 'detection failed' : 'queued';
    const genDot = \`<span class="gen-dot \${dotClass}" title="\${dotTitle}"></span>\`;
    btn.innerHTML = modDot + '<span style="flex:1">' + filename + '</span>' + genDot;
    btn.addEventListener('click', () => selectImage(i));
    list.appendChild(btn);
  });
}

// ---- Data panel ----
function updateDataPanel() {
  const ann = getAnnotation(currentIdx);
  let html = '';

  ann.targets.forEach((target, ti) => {
    const isActiveTarget = ti === currentTargetIdx;
    const headerStyle = isActiveTarget ? 'color:#4a9eff;font-weight:bold' : 'color:#888;cursor:pointer';
    html += \`<div class="target-header" data-ti="\${ti}" style="\${headerStyle};font-size:0.72rem;padding:2px 0">Target \${ti+1} · \${target.paperBoundary.length} boundary pts</div>\`;
    html += '<table><thead><tr><th>Ring set</th><th>Ring</th><th>pts</th><th>cx</th><th>cy</th></tr></thead><tbody>';
    target.ringSets.forEach((rings, si) => {
      const isActiveRS = isActiveTarget && si === currentRingSetIdx;
      rings.forEach((ring, ri) => {
        if (!ring.points) return;
        const cx = (ring.points.reduce((s, p) => s + p[0], 0) / ring.points.length).toFixed(0);
        const cy = (ring.points.reduce((s, p) => s + p[1], 0) / ring.points.length).toFixed(0);
        const rsLabel = ri === 0 ? \`RS\${si+1}\` : '';
        const rowStyle = isActiveRS ? 'color:#ccc' : 'color:#666;cursor:pointer';
        html += \`<tr class="rs-row" data-ti="\${ti}" data-si="\${si}" style="\${rowStyle}"><td>\${rsLabel}</td><td>R\${ri}</td><td>\${ring.points.length}</td><td>\${cx}</td><td>\${cy}</td></tr>\`;
      });
    });
    html += '</tbody></table>';
  });

  if (ann.arrows.length > 0) {
    html += \`<table><thead><tr><th>Arrow</th><th>tip</th><th>score</th></tr></thead><tbody>\`;
    ann.arrows.forEach((arrow, ai) => {
      const s = arrow.score;
      const scoreLabel = s === 0 ? 'M' : (s === null || s === undefined) ? '?' : String(s);
      html += \`<tr>
        <td>A\${ai}</td>
        <td>(\${Math.round(arrow.tip[0])},\${Math.round(arrow.tip[1])})</td>
        <td class="score-cell" data-ai="\${ai}">\${scoreLabel}</td>
      </tr>\`;
    });
    html += '</tbody></table>';
  } else {
    html += \`<div style="color:#555;font-size:0.72rem;margin-top:4px">No arrows annotated</div>\`;
  }

  document.getElementById('data-table').innerHTML = html;

  document.querySelectorAll('#data-table .score-cell').forEach(cell => {
    cell.addEventListener('click', () => {
      const ai = parseInt(cell.getAttribute('data-ai'), 10);
      pickerContext = { type: 'edit', ai };
      const handle = document.querySelector('[data-handle="arrow_tip"][data-ai="' + ai + '"]');
      let cx, cy;
      if (handle) { const r = handle.getBoundingClientRect(); cx = (r.left + r.right) / 2; cy = (r.top + r.bottom) / 2; }
      showScorePicker(\`Edit A\${ai} score:\`, undefined, cx, cy);
    });
  });

  document.querySelectorAll('#data-table .target-header').forEach(el => {
    el.addEventListener('click', () => {
      const ti = parseInt(el.getAttribute('data-ti'), 10);
      if (ti !== currentTargetIdx) { currentTargetIdx = ti; currentRingSetIdx = 0; render(); }
    });
  });

  document.querySelectorAll('#data-table .rs-row').forEach(row => {
    row.addEventListener('click', () => {
      const ti = parseInt(row.getAttribute('data-ti'), 10);
      const si = parseInt(row.getAttribute('data-si'), 10);
      if (ti !== currentTargetIdx || si !== currentRingSetIdx) {
        currentTargetIdx = ti; currentRingSetIdx = si; render();
      }
    });
  });
}

// ---- Save to DB ----
function showSaveMsg(text) {
  const el = document.getElementById('save-msg');
  el.textContent = text;
  el.style.display = 'block';
  clearTimeout(el._hideTimer);
  el._hideTimer = setTimeout(() => { el.style.display = 'none'; }, 4000);
}

// ms=0 → persistent until next call; ms>0 → auto-hide after ms
function showStatusMsg(text, ms = 0) {
  const el = document.getElementById('save-msg');
  el.textContent = text;
  el.style.display = 'block';
  clearTimeout(el._hideTimer);
  if (ms > 0) el._hideTimer = setTimeout(() => { el.style.display = 'none'; }, ms);
}

async function save() {
  console.log(\`[save] clicked — modified: [\${store.modified.join(', ') || 'none'}]\`);
  const out = {};
  for (const filename of Object.keys(store.annotations)) {
    const ann = store.annotations[filename];
    out[filename] = { targets: ann.targets || [], arrows: ann.arrows || [] };
  }
  const total = Object.keys(out).length;
  console.log(\`[save] sending \${total} annotation(s)\`);
  for (const [filename, ann] of Object.entries(out)) {
    const modified = store.modified.includes(filename);
    console.log(\`[save]   \${modified ? '* ' : '  '}\${filename}: targets=\${ann.targets?.length ?? 0} arrows=\${ann.arrows.length}\`);
  }
  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(out),
    });
    console.log(\`[save] response HTTP \${res.status}\`);
    if (!res.ok) throw new Error(\`HTTP \${res.status}\`);
    const body = await res.json();
    console.log(\`[save] server response:\`, body);
  } catch (e) {
    console.error('[save] failed:', e);
    alert('Save failed: ' + e);
    return;
  }
  store.modified = [];
  updateImageList();
}

// ---- Reset functions ----
function resetBoundaries() {
  const detected = getDetected(currentIdx);
  const ann = getAnnotation(currentIdx);
  detected.targets.forEach((dt, t) => {
    if (ann.targets[t]) {
      ann.targets[t].paperBoundary = dt.paperBoundary.map(p => [p[0], p[1]]);
    }
  });
  markModified(currentIdx);
  updateImageList();
  render();
}

function resetRings() {
  const detected = getDetected(currentIdx);
  const ann = getAnnotation(currentIdx);
  detected.targets.forEach((dt, t) => {
    if (ann.targets[t]) {
      ann.targets[t].ringSets = dt.ringSets.map(rs => rs.map(r => ({ points: r.points.map(p => [p[0], p[1]]) })));
    }
  });
  markModified(currentIdx);
  updateImageList();
  render();
}

function resetArrows() {
  const ann = getAnnotation(currentIdx);
  ann.arrows = [];
  markModified(currentIdx);
  updateImageList();
  render();
}

function resetCurrent() {
  const filename = IMAGES[currentIdx];
  delete imageDataCache[filename];
  delete store.annotations[filename];
  store.modified = store.modified.filter(f => f !== filename);
  currentTargetIdx = 0;
  currentRingSetIdx = 0;
  addArrowMode = 'idle';
  addRingSetMode = 'idle';
  pendingTip = null;
  hideScorePicker();
  document.getElementById('btn-add-arrow').classList.remove('active-mode');
  document.getElementById('btn-add-ringset').classList.remove('active-mode');
  updateImageList();
  renderLoading('Recomputing\u2026');
  fetch('/api/recompute/' + encodeURIComponent(filename), { method: 'POST' }).catch(() => {});
}

// ---- Delete image (AW-7) ----
async function deleteCurrentImage() {
  const filename = IMAGES[currentIdx];
  if (!confirm(\`Delete "\${filename}" and all its annotation data? This cannot be undone.\`)) return;
  try {
    const res = await fetch('/api/image/' + encodeURIComponent(filename), { method: 'DELETE' });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || 'HTTP ' + res.status);
    }
  } catch (e) {
    alert('Delete failed: ' + e);
    return;
  }
  IMAGES.splice(currentIdx, 1);
  delete imageDataCache[filename];
  delete store.annotations[filename];
  store.modified = store.modified.filter(f => f !== filename);
  delete generationStatus[filename];
  if (IMAGES.length === 0) {
    currentIdx = 0;
    document.getElementById('svg-container').innerHTML =
      '<div class="loading-wrap"><div style="color:#666">No images remaining.</div></div>';
    document.getElementById('data-table').innerHTML = '';
    updateImageList();
    return;
  }
  currentIdx = Math.min(currentIdx, IMAGES.length - 1);
  updateImageList();
  selectImage(currentIdx);
}

// ---- Keyboard handler ----
document.addEventListener('keydown', (e) => {
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

  if (e.key === 'Tab') {
    e.preventDefault();
    setViewMode(viewMode === 'annotated' ? 'generated' : 'annotated');
    return;
  }

  if (e.key === 'Escape') {
    if (addRingSetMode === 'place-center') {
      addRingSetMode = 'idle';
      document.getElementById('btn-add-ringset').classList.remove('active-mode');
      render();
      return;
    }
    if (addArrowMode === 'score-input' && pickerContext) {
      if (pickerContext.type === 'new') {
        const ann = getAnnotation(currentIdx);
        ann.arrows.push({
          tip:  [Math.round(pendingTip[0]),  Math.round(pendingTip[1])],
          nock: null,
          score: null,
        });
        pendingTip = null;
        addArrowMode = 'idle';
        document.getElementById('btn-add-arrow').classList.remove('active-mode');
        hideScorePicker();
        markModified(currentIdx);
        updateImageList();
        render();
      } else {
        hideScorePicker();
        addArrowMode = 'idle';
      }
    } else if (addArrowMode === 'place-tip') {
      pendingTip = null;
      addArrowMode = 'idle';
      document.getElementById('btn-add-arrow').classList.remove('active-mode');
      render();
    }
    return;
  }

  if (addArrowMode === 'score-input') {
    let score = undefined;
    if (e.key === 'x' || e.key === 'X') score = 'X';
    else if (e.key === 'm' || e.key === 'M') score = 0;
    else if (e.key >= '1' && e.key <= '9') score = parseInt(e.key, 10);
    if (score !== undefined) { onScoreSelect(score); return; }
  }

  if (e.key === 'b' || e.key === 'B') {
    if (currentIdx > 0) selectImage(currentIdx - 1);
    return;
  }
  if (e.key === 'n' || e.key === 'N') {
    if (currentIdx < IMAGES.length - 1) { save(); selectImage(currentIdx + 1); }
    return;
  }

  if ((e.key === 'a' || e.key === 'A') && addArrowMode !== 'score-input' && viewMode === 'annotated') {
    if (addArrowMode === 'idle') {
      addArrowMode = 'place-tip';
      document.getElementById('btn-add-arrow').classList.add('active-mode');
      render();
    } else if (addArrowMode === 'place-tip') {
      addArrowMode = 'idle';
      document.getElementById('btn-add-arrow').classList.remove('active-mode');
      render();
    }
    return;
  }

  if ((e.key === 'r' || e.key === 'R') && viewMode === 'annotated') {
    if (addRingSetMode === 'idle') {
      addRingSetMode = 'place-center';
      document.getElementById('btn-add-ringset').classList.add('active-mode');
      render();
    } else if (addRingSetMode === 'place-center') {
      addRingSetMode = 'idle';
      document.getElementById('btn-add-ringset').classList.remove('active-mode');
      render();
    }
    return;
  }
});

// ---- Wire up ----
document.getElementById('btn-save').addEventListener('click', save);
document.getElementById('btn-reset').addEventListener('click', resetCurrent);
document.getElementById('btn-reset-boundaries').addEventListener('click', resetBoundaries);
document.getElementById('btn-reset-rings').addEventListener('click', resetRings);
document.getElementById('btn-reset-arrows').addEventListener('click', resetArrows);
document.getElementById('btn-remove-ringset').addEventListener('click', removeActiveRingSet);
document.getElementById('btn-delete').addEventListener('click', deleteCurrentImage);
document.getElementById('chk-rings').addEventListener('change', render);
document.getElementById('chk-boundary').addEventListener('change', render);
document.getElementById('chk-arrows').addEventListener('change', render);
document.getElementById('btn-view-annotated').addEventListener('click', () => setViewMode('annotated'));
document.getElementById('btn-view-generated').addEventListener('click', () => setViewMode('generated'));

['all', 'annotated', 'unannotated'].forEach(f => {
  document.getElementById('filter-' + f).addEventListener('click', () => {
    imageFilter = f;
    document.querySelectorAll('#img-filter button').forEach(b => b.classList.remove('active'));
    document.getElementById('filter-' + f).classList.add('active');
    updateImageList();
  });
});

document.getElementById('btn-add-arrow').addEventListener('click', () => {
  if (viewMode === 'generated') return;
  if (addArrowMode === 'idle') {
    addArrowMode = 'place-tip';
    document.getElementById('btn-add-arrow').classList.add('active-mode');
    render();
  } else if (addArrowMode === 'place-tip') {
    pendingTip = null;
    addArrowMode = 'idle';
    document.getElementById('btn-add-arrow').classList.remove('active-mode');
    render();
  }
});

document.getElementById('btn-add-ringset').addEventListener('click', () => {
  if (viewMode === 'generated') return;
  if (addRingSetMode === 'idle') {
    addRingSetMode = 'place-center';
    document.getElementById('btn-add-ringset').classList.add('active-mode');
    render();
  } else if (addRingSetMode === 'place-center') {
    addRingSetMode = 'idle';
    document.getElementById('btn-add-ringset').classList.remove('active-mode');
    render();
  }
});

document.querySelectorAll('#score-picker button[data-score]').forEach(btn => {
  btn.addEventListener('click', () => {
    const raw = btn.getAttribute('data-score');
    if (raw === null) return;
    const score = raw === 'X' ? 'X' : parseInt(raw, 10);
    if (typeof score === 'number' && isNaN(score)) return;
    onScoreSelect(score);
  });
});
document.getElementById('score-skip').addEventListener('click', () => onScoreSelect(null));
document.getElementById('score-picker').addEventListener('mousedown', (e) => e.stopPropagation());

// ---- Collapsible data panel (AW-3) ----
(function() {
  const panel = document.getElementById('data-panel');
  const btn   = document.getElementById('btn-collapse-data');
  function setCollapsed(v) {
    panel.classList.toggle('collapsed', v);
    btn.textContent = v ? '▶' : '▼';
    try { localStorage.setItem('dataPanelCollapsed', v ? '1' : ''); } catch {}
  }
  const saved = (() => { try { return localStorage.getItem('dataPanelCollapsed'); } catch { return null; } })();
  if (saved === '1') setCollapsed(true);
  btn.addEventListener('click', () => setCollapsed(!panel.classList.contains('collapsed')));
  document.getElementById('data-panel-header').addEventListener('click', (e) => {
    if (e.target === btn) return;
    setCollapsed(!panel.classList.contains('collapsed'));
  });
})();

// ---- SSE subscription (AW-5) ----
(function() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'status' && msg.filename) {
        generationStatus[msg.filename] = msg.state;
        const idx = IMAGES.indexOf(msg.filename);
        if (idx >= 0) updateImageList();
        if (msg.state === 'ready' && idx === currentIdx && !imageDataCache[msg.filename]) {
          selectImage(idx);
        }
      } else if (msg.type === 'new_image' && msg.filename) {
        if (!IMAGES.includes(msg.filename)) {
          IMAGES.push(msg.filename);
          generationStatus[msg.filename] = 'queued';
          updateImageList();
        }
      } else if (msg.type === 'removed' && msg.filename) {
        const idx = IMAGES.indexOf(msg.filename);
        if (idx >= 0) {
          IMAGES.splice(idx, 1);
          delete imageDataCache[msg.filename];
          delete store.annotations[msg.filename];
          store.modified = store.modified.filter(f => f !== msg.filename);
          delete generationStatus[msg.filename];
          if (IMAGES.length === 0) {
            currentIdx = 0;
            document.getElementById('svg-container').innerHTML =
              '<div class="loading-wrap"><div style="color:#666">No images remaining.</div></div>';
            document.getElementById('data-table').innerHTML = '';
            updateImageList();
          } else if (currentIdx === idx) {
            currentIdx = Math.min(idx, IMAGES.length - 1);
            updateImageList();
            selectImage(currentIdx);
          } else {
            if (currentIdx > idx) currentIdx--;
            updateImageList();
          }
        }
      }
    } catch {}
  };
  es.onerror = () => {
    console.warn('[SSE] Connection error — will reconnect automatically.');
  };
})();

// ---- Init ----
fetch('/api/generation-status')
  .then(r => r.json())
  .then(data => { generationStatus = data; })
  .catch(() => {})
  .finally(() => {
    fetch('/api/annotations')
      .then(r => r.json())
      .then(data => {
        for (const [filename, ann] of Object.entries(data)) {
          store.annotations[filename] = {
            targets: ann.targets || [],
            arrows:  ann.arrows  || [],
          };
        }
      })
      .catch(() => {})
      .finally(() => {
        fetch('/api/stale-images')
          .then(r => r.json())
          .then(({ stale }) => { staleImages = new Set(stale); })
          .catch(() => {})
          .finally(() => { updateImageList(); selectImage(0); });
      });
  });
</script>
</body>
</html>`;
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
    CREATE TABLE IF NOT EXISTS annotations (
      filename       TEXT PRIMARY KEY,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      arrows         JSONB NOT NULL DEFAULT '[]',
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await db.query(`
    CREATE TABLE IF NOT EXISTS generated (
      filename       TEXT PRIMARY KEY,
      algorithm_hash TEXT NOT NULL,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      arrows         JSONB NOT NULL DEFAULT '[]',
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

  type GenState = 'ready' | 'queued' | 'computing' | 'error';
  const generationStatus = new Map<string, GenState>(
    filenames.map(f => [f, inGenerated.get(f) === currentHash ? 'ready' : 'queued']),
  );
  const sseClients = new Set<http.ServerResponse>();
  function broadcastSSE(data: object) {
    const msg = `data: ${JSON.stringify(data)}\n\n`;
    for (const client of sseClients) {
      try { client.write(msg); } catch { sseClients.delete(client); }
    }
  }

  const imageCache = new Map<string, CachedImageData>();
  // Deduplicates concurrent /api/image/ requests for the same file so detection
  // runs at most once.  Second caller awaits the first caller's promise.
  const pendingImageRequests = new Map<string, Promise<void>>();

  const html = generateHtml(filenames);

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
      if (!imageCache.has(filename)) {
        const imgPath = path.join(IMAGES_DIR, filename);
        if (!fs.existsSync(imgPath)) { respond(404, '{"error":"not found"}'); return; }

        if (pendingImageRequests.has(filename)) {
          await pendingImageRequests.get(filename)!;
        } else {
          let resolveDetect!: () => void, rejectDetect!: (e: unknown) => void;
          const detectPromise = new Promise<void>(
            (res, rej) => { resolveDetect = res; rejectDetect = rej; },
          );
          pendingImageRequests.set(filename, detectPromise);
          try {

        const hashInGen = inGenerated.get(filename);
        let isReady = hashInGen === currentHash;
        console.log(`image request: ${filename}  hashInGen=${hashInGen ?? 'null'}  isReady=${isReady}`);

        let base64: string;
        let width: number;
        let height: number;
        let detectedTargets: TargetData[];
        let detectedArrows: { tip: [number, number]; nock: [number, number] | null }[];

        if (isReady) {
          // Fast path (AW-1): read from generated table
          console.log(`  fast path: loading ${filename}…`);
          ({ base64, width, height } = await loadImageBase64(imgPath));
          const { rows } = await db.query(
            'SELECT rings, paper_boundary, arrows FROM generated WHERE filename = $1',
            [filename],
          );
          detectedTargets = dbToTargets(rows[0]?.paper_boundary ?? null, rows[0]?.rings ?? null);
          detectedArrows  = rows[0]?.arrows ?? [];

          // AW-2: validity check — fall back to slow path if data is corrupt
          if (!isValidDetected(detectedTargets)) {
            logEvent('warn', 'invalid_generated', filename, 'null coordinates in stored data — recomputing');
            logEvent('info', 'fallback_slow_path', filename, 'invalid cache');
            await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
            inGenerated.delete(filename);
            generationStatus.set(filename, 'queued');
            isReady = false;
          }
        }

        if (!isReady) {
          // Slow path: run detection synchronously
          generationStatus.set(filename, 'computing');
          broadcastSSE({ type: 'status', filename, state: 'computing' });
          const entry = await processImage(imgPath);
          const ok = entry.result.success;
          console.log(`  ${filename} ... ${ok ? 'ok' : `FAILED: ${(entry.result as any).error}`}`);
          if (!ok) logEvent('error', 'detection_failed', filename, (entry.result as any).error ?? '');
          base64 = entry.base64;
          width = entry.width;
          height = entry.height;

          const rawRings    = ok ? entry.result.rings : [];
          const rawBoundary = clampBoundary(
            ok && entry.result.paperBoundary ? entry.result.paperBoundary.points : null,
            entry.width ?? 0, entry.height ?? 0,
          );
          detectedArrows = entry.detectedArrows;

          const { boundary: dbBoundary, rings: dbRings } = wrapSingleTarget(rawRings, rawBoundary);
          detectedTargets = dbToTargets(dbBoundary, dbRings);

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
             JSON.stringify(detectedArrows), entry.width ?? null, entry.height ?? null],
          );
          inGenerated.set(filename, currentHash);
          generationStatus.set(filename, 'ready');
          broadcastSSE({ type: 'status', filename, state: 'ready' });

          // Seed / heal annotation boundary. The DO UPDATE fires only when the
          // stored boundary is absent or degenerate (all-zero), so user edits
          // to rings and arrows are never overwritten by a re-detection.
          await db.query(
            `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
             VALUES ($1, $2, $3, '[]')
             ON CONFLICT (filename) DO UPDATE
               SET paper_boundary = EXCLUDED.paper_boundary,
                   updated_at     = NOW()
               WHERE annotations.paper_boundary IS NULL
                  OR annotations.paper_boundary = '[]'::jsonb
                  OR ((annotations.paper_boundary->0->0->0)::float = 0
                 AND (annotations.paper_boundary->0->0->1)::float = 0)`,
            [filename, JSON.stringify(dbBoundary), JSON.stringify(dbRings)],
          );
          inAnnotations.add(filename);
        }

        imageCache.set(filename, {
          base64,
          width,
          height,
          detected: { targets: detectedTargets, arrows: detectedArrows },
        });
          resolveDetect();
          } catch (detErr) { rejectDetect(detErr); throw detErr; }
          finally { pendingImageRequests.delete(filename); }
        }
      }

      respond(200, JSON.stringify(imageCache.get(filename)!));
      } catch (err) {
        console.error('Error processing image:', err);
        respond(500, JSON.stringify({ error: String(err) }));
      }

    } else if (req.method === 'GET' && req.url === '/api/annotations') {
      // LEFT JOIN with generated so that degenerate (all-zero) annotation
      // boundaries are transparently replaced with the detected boundary.
      const { rows } = await db.query(`
        SELECT a.filename,
          CASE WHEN a.paper_boundary IS NULL
                 OR a.paper_boundary = '[]'::jsonb
                 OR ((a.paper_boundary->0->0->0)::float = 0
                AND (a.paper_boundary->0->0->1)::float = 0)
               THEN COALESCE(g.paper_boundary, '[]'::jsonb)
               ELSE a.paper_boundary
          END AS paper_boundary,
          a.rings, a.arrows
        FROM annotations a
        LEFT JOIN generated g ON a.filename = g.filename
      `);
      const out: Record<string, unknown> = {};
      const toDelete: string[] = [];
      for (const row of rows) {
        const targets = dbToTargets(row.paper_boundary, row.rings);
        const arrows  = row.arrows ?? [];
        if (!isValidAnnotation(targets, arrows)) {
          toDelete.push(row.filename);
          continue;
        }
        out[row.filename] = { targets, arrows };
      }
      if (toDelete.length > 0) {
        await db.query(
          `DELETE FROM annotations WHERE filename = ANY($1)`,
          [toDelete],
        );
        for (const f of toDelete) {
          inAnnotations.delete(f);
          logEvent('info', 'invalid_annotation_deleted', f, 'removed by /api/annotations filter');
        }
      }
      respond(200, JSON.stringify(out));

    } else if (req.method === 'GET' && req.url?.startsWith('/api/annotation/')) {
      const filename = decodeURIComponent(req.url.slice('/api/annotation/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      const { rows } = await db.query(
        `SELECT
          CASE WHEN a.paper_boundary IS NULL
                 OR a.paper_boundary = '[]'::jsonb
                 OR ((a.paper_boundary->0->0->0)::float = 0
                AND (a.paper_boundary->0->0->1)::float = 0)
               THEN COALESCE(g.paper_boundary, '[]'::jsonb)
               ELSE a.paper_boundary
          END AS paper_boundary,
          a.rings, a.arrows
         FROM annotations a
         LEFT JOIN generated g ON a.filename = g.filename
         WHERE a.filename = $1`,
        [filename],
      );
      if (rows.length === 0) { respond(404, '{"error":"not found"}'); return; }
      respond(200, JSON.stringify({
        targets: dbToTargets(rows[0].paper_boundary, rows[0].rings),
        arrows:  rows[0].arrows ?? [],
      }));

    } else if (req.method === 'POST' && req.url?.startsWith('/api/recompute/')) {
      const filename = decodeURIComponent(req.url.slice('/api/recompute/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      imageCache.delete(filename);
      inGenerated.delete(filename);
      inAnnotations.delete(filename);
      await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
      await db.query('DELETE FROM annotations WHERE filename = $1', [filename]);
      generationStatus.set(filename, 'computing');
      broadcastSSE({ type: 'status', filename, state: 'computing' });
      respond(202, '{"status":"computing"}');
      (async () => {
        try {
          const imgPath = path.join(IMAGES_DIR, filename);
          const entry = await processImage(imgPath);
          const ok = entry.result.success;
          if (!ok) logEvent('error', 'detection_failed', filename, (entry.result as any).error ?? '');

          const rawRings    = ok ? entry.result.rings : [];
          const rawBoundary = clampBoundary(
            ok && entry.result.paperBoundary ? entry.result.paperBoundary.points : null,
            entry.width ?? 0, entry.height ?? 0,
          );
          const detectedArrows = entry.detectedArrows;
          const { boundary: dbBoundary, rings: dbRings } = wrapSingleTarget(rawRings, rawBoundary);

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
             JSON.stringify(detectedArrows), entry.width ?? null, entry.height ?? null],
          );
          await db.query(
            `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
             VALUES ($1, $2, $3, '[]')`,
            [filename, JSON.stringify(dbBoundary), JSON.stringify(dbRings)],
          );
          inGenerated.set(filename, currentHash);
          inAnnotations.add(filename);
          generationStatus.set(filename, 'ready');
          broadcastSSE({ type: 'status', filename, state: 'ready' });
        } catch (err) {
          console.error('Recompute error:', err);
          logEvent('error', 'recompute_failed', filename, String(err));
          generationStatus.set(filename, 'error');
          broadcastSSE({ type: 'status', filename, state: 'error' });
        }
      })();

    } else if (req.method === 'POST' && req.url === '/api/save') {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
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
              paperBoundary: clampBoundary(t.paperBoundary ?? [], w, h) ?? [],
              ringSets:      t.ringSets ?? [],
            }));
            const { boundary: dbBoundary, rings: dbRings } = targetsToDB(targets);
            const arrows = ann.arrows ?? [];
            if (!isValidAnnotation(targets, arrows)) {
              console.log(`[save]   SKIP ${filename}: annotation invalid (targets=${targets.length}, arrows=${arrows.length})`);
              logEvent('warn', 'save-skipped-invalid', filename, `targets=${targets.length} arrows=${arrows.length}`);
              continue;
            }
            await db.query(
              `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (filename) DO UPDATE
                 SET paper_boundary = EXCLUDED.paper_boundary,
                     rings          = EXCLUDED.rings,
                     arrows         = EXCLUDED.arrows,
                     updated_at     = NOW()`,
              [filename, JSON.stringify(dbBoundary), JSON.stringify(dbRings), JSON.stringify(arrows)],
            );
            console.log(`[save]   OK ${filename}: targets=${targets.length}, arrows=${arrows.length}`);
            logEvent('info', 'save-ok', filename, `targets=${targets.length} arrows=${arrows.length}`);
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
      req.on('end', async () => {
        try {
          const { filename, cx, cy } = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
          const imgPath = path.join(IMAGES_DIR, filename);
          const { rgba, width, height } = await loadImageNode(imgPath);
          const t = findRingSetFromCenter(rgba, width, height, cx, cy);
          const boundary = clampBoundary(t.paperBoundary.points, width, height) ?? [];
          respond(200, JSON.stringify({ rings: t.rings, paperBoundary: boundary }));
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

  // Shared background processing function
  async function processFileInBackground(filename: string): Promise<void> {
    if (inGenerated.get(filename) === currentHash) return;
    const imgPath = path.join(IMAGES_DIR, filename);
    generationStatus.set(filename, 'computing');
    broadcastSSE({ type: 'status', filename, state: 'computing' });
    try {
      const result = await runWorkerProcess(imgPath);
      if (!result.success) {
        logEvent('error', 'detection_failed', filename, result.error ?? '');
      }
      const { rings, arrows, width: imgW, height: imgH } = result;
      const rawBoundary = clampBoundary(result.paperBoundary ?? null, imgW ?? 0, imgH ?? 0);
      const { boundary: dbBoundary, rings: dbRings } = wrapSingleTarget(rings, rawBoundary);

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
         JSON.stringify(dbBoundary), JSON.stringify(dbRings), JSON.stringify(arrows),
         imgW ?? null, imgH ?? null],
      );
      inGenerated.set(filename, currentHash);
      generationStatus.set(filename, 'ready');
      broadcastSSE({ type: 'status', filename, state: 'ready' });

      await db.query(
        `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
         VALUES ($1, $2, $3, '[]')
         ON CONFLICT (filename) DO UPDATE
           SET paper_boundary = EXCLUDED.paper_boundary,
               updated_at     = NOW()
           WHERE annotations.paper_boundary IS NULL
              OR annotations.paper_boundary = '[]'::jsonb
              OR (annotations.paper_boundary->0->0->>'0')::float = 0
             AND (annotations.paper_boundary->0->0->>'1')::float = 0`,
        [filename, JSON.stringify(dbBoundary), JSON.stringify(dbRings)],
      );
      inAnnotations.add(filename);
      console.log(`  [bg] ${filename} … ok`);
    } catch (err) {
      const msg = String(err);
      console.error(`  [bg] ${filename} … FAILED: ${msg}`);
      logEvent('error', 'detection_failed', filename, msg);
      generationStatus.set(filename, 'queued');
      broadcastSSE({ type: 'status', filename, state: 'queued' });
    }
  }

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Annotation tool: http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop.');
    if (!process.env.NO_BROWSER) require('child_process').exec(`open http://localhost:${PORT}`);

    const toProcess = filenames.filter(f => inGenerated.get(f) !== currentHash);
    if (toProcess.length > 0) {
      console.log(`Background queue: ${toProcess.length} image(s) to process…`);
      (async () => {
        for (const filename of toProcess) await processFileInBackground(filename);
        console.log('Background queue complete.');
      })().catch(err => console.error('Background queue error:', err));
    }

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
        processFileInBackground(name).catch(err => console.error(`Error processing new image ${name}:`, err));
      }, 500);
    });
  });
}

main().catch(err => { console.error(err); process.exit(1); });
