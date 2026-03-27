import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
import { Pool } from 'pg';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, ArcheryResult } from '../src/targetDetection';
import { findArrows, ArrowDetection } from '../src/arrowDetection';
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

interface CachedImageData {
  base64: string;
  width: number;
  height: number;
  detected: {
    rings: SplineRing[];
    paperBoundary: [number, number][] | null;
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

function isValidDetected(
  rings: SplineRing[],
  boundary: [number, number][] | null,
): boolean {
  const ringsOk = rings.every(r => r.points?.every(p => p[0] != null && p[1] != null));
  const boundaryOk = boundary == null || boundary.every(p => p[0] != null && p[1] != null);
  return ringsOk && boundaryOk;
}

function isValidAnnotation(ann: {
  paperBoundary: [number, number][] | null;
  rings: any[];
  arrows: any[];
}): boolean {
  return (
    ann.paperBoundary != null && ann.paperBoundary.length >= 3 &&
    ann.rings.length == 10 &&
    ann.rings.every((r: any) => Array.isArray(r.points) && r.points.length >= 3)
  );
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
  success: boolean;
  error?: string;
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
    path.resolve(__dirname, '../src/arrowDetection.ts'),
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
  detectedArrows: ArrowDetection[];
}

async function processImage(imgPath: string): Promise<ProcessedImage> {
  const filename = path.basename(imgPath);
  console.log(`  [1/4] loadImageNode ${filename}…`);
  const { rgba, width, height } = await loadImageNode(imgPath);
  console.log(`  [2/4] loadImageBase64 ${filename} (${width}×${height})…`);
  const { base64 } = await loadImageBase64(imgPath);
  console.log(`  [3/4] findTarget ${filename}…`);
  const result = findTarget(rgba, width, height);
  console.log(`  [4/4] findArrows ${filename} (success=${result.success})…`);
  const detectedArrows = findArrows(rgba, width, height, result);
  console.log(`  done: ${filename} — ${detectedArrows.length} arrows`);
  return { filename, base64, width, height, result, detectedArrows };
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
    #save-msg { font-size: 0.75rem; color: #e07070; padding: 4px 0 0; display: none; line-height: 1.4; }

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

    #score-picker { display: none; gap: 4px; align-items: center; flex-wrap: wrap; width: 100%; padding-top: 6px; }
    #score-picker .sp-label { font-size: 0.78rem; color: #FF8C00; white-space: nowrap; margin-right: 2px; }
    #score-picker button { padding: 3px 7px; font-size: 0.78rem; border: 1px solid #555; border-radius: 3px; background: #2a2a2a; color: #ddd; cursor: pointer; }
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

    /* AW-4: suggested score highlight */
    #score-picker button.suggested { outline: 2px solid #4a9eff; outline-offset: 1px; }

    #main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: auto; background: #111; padding: 16px; }
    #svg-container { position: relative; }
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
      <button id="btn-reset">Reset image</button>
      <button id="btn-add-arrow">Add arrow (A)</button>
      <button id="btn-reset-all" class="danger">Reset all</button>
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
    <label><input type="checkbox" id="chk-handles" checked/> Handles</label>
    <label><input type="checkbox" id="chk-arrows" checked/> Arrows</label>
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
    <div id="legend">
      Ctrl+click boundary: add vertex · Alt+click ring: add point · Shift+click vertex/ring pt/arrow: remove · A: add arrow
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
let drag = null;
let addArrowMode = 'idle'; // 'idle' | 'place-tip' | 'place-nock' | 'score-input'
let pendingTip = null;
let pendingNock = null;
let pickerContext = null; // { type: 'new' } | { type: 'edit', ai: number } | null
let imageDataCache = {}; // filename -> { base64, width, height, detected }
let viewMode = 'annotated'; // 'annotated' | 'generated'
let imageFilter = 'all'; // 'all' | 'annotated' | 'unannotated'

// ---- Overlay toggles ----
function showRings()    { return document.getElementById('chk-rings').checked; }
function showBoundary() { return document.getElementById('chk-boundary').checked; }
function showHandles()  { return document.getElementById('chk-handles').checked; }
function showArrows()   { return document.getElementById('chk-arrows').checked; }

// ---- Annotation helpers ----
function getDetected(idx) {
  const data = imageDataCache[IMAGES[idx]];
  if (!data) return { paperBoundary: null, rings: [] };
  return {
    paperBoundary: data.detected.paperBoundary ? data.detected.paperBoundary.map(p => [p[0], p[1]]) : null,
    rings: data.detected.rings,
  };
}

function getAnnotation(idx) {
  const filename = IMAGES[idx];
  if (!store.annotations[filename]) store.annotations[filename] = { ...getDetected(idx), arrows: [] };
  const ann = store.annotations[filename];
  if (!ann.arrows) ann.arrows = [];
  return ann;
}

function isAnnotationValid(ann) {
  return ann.paperBoundary != null && ann.paperBoundary.length >= 3 &&
         ann.rings.length > 0 && ann.arrows.length > 0;
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
function showScorePicker(label, suggested) {
  const picker = document.getElementById('score-picker');
  picker.style.display = 'flex';
  document.getElementById('score-prompt').textContent = label || 'Score:';
  // Highlight suggested button
  picker.querySelectorAll('button[data-score]').forEach(btn => {
    const raw = btn.getAttribute('data-score');
    const val = raw === 'X' ? 'X' : parseInt(raw, 10);
    btn.classList.toggle('suggested', suggested !== undefined && val === suggested);
  });
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
      nock: [Math.round(pendingNock[0]), Math.round(pendingNock[1])],
      score,
    });
    pendingTip = null;
    pendingNock = null;
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
  // Exit add-arrow mode when switching to generated (read-only)
  if (mode === 'generated' && addArrowMode !== 'idle') {
    pendingTip = null; pendingNock = null;
    addArrowMode = 'idle';
    hideScorePicker();
    document.getElementById('btn-add-arrow').classList.remove('active-mode');
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
  updateImageList();
  const filename = IMAGES[idx];
  console.log('[selectImage] clicked:', filename, 'cached:', !!imageDataCache[filename], 'stale:', staleImages.has(filename));
  if (imageDataCache[filename]) {
    render();
    return;
  }

  // Show loading immediately using the stale hint we already have from /api/stale-images
  let computing = staleImages.has(filename);
  renderLoading(computing ? 'Computing rings\u2026 0s' : 'Loading\u2026');

  // Confirm with a lightweight status check (fast DB query), update message if hint was wrong
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
    console.log('[selectImage] done, detected rings:', imageDataCache[filename]?.detected?.rings?.length, 'arrows:', imageDataCache[filename]?.detected?.arrows?.length);
    staleImages.delete(filename);
    // Reload annotation from DB if it was cleared (e.g. after recompute reset)
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
  if (!data) return; // still loading
  const ann = getAnnotation(currentIdx);

  const isGenerated = viewMode === 'generated';
  const rings = isGenerated ? data.detected.rings : ann.rings;
  const boundary = isGenerated
    ? (data.detected.paperBoundary ? data.detected.paperBoundary.map(p => [p[0], p[1]]) : null)
    : ann.paperBoundary;

  const W = data.width, H = data.height;
  const showR = showRings(), showB = showBoundary(), showH = showHandles() && !isGenerated, showA = showArrows();

  let svgContent = '';

  // Boundary polygon
  if (boundary && showB && boundary.every(p => p[0] != null && p[1] != null)) {
    const pts = boundary.map(p => \`\${p[0].toFixed(1)},\${p[1].toFixed(1)}\`).join(' ');
    svgContent += \`<polygon points="\${pts}" fill="none" stroke="#00FF88" stroke-width="3" stroke-dasharray="12 6" opacity="0.85"/>\`;
  }

  // Ring splines — draw outermost first
  if (showR && rings.length > 0) {
    for (let i = rings.length - 1; i >= 0; i--) {
      const ring = rings[i];
      if (!ring.points || ring.points.length < 3) continue;
      if (ring.points.some(p => p[0] == null || p[1] == null)) continue; // skip corrupt points
      const color = RING_COLORS[i] || '#FFFFFF';
      const d = splineToPath(ring.points);
      if (i >= 8) {
        svgContent += \`<path d="\${d}" fill="none" stroke="#000" stroke-width="4"/>\`;
        svgContent += \`<path d="\${d}" fill="none" stroke="\${color}" stroke-width="2"/>\`;
      } else if (i >= 6) {
        svgContent += \`<path d="\${d}" fill="none" stroke="#888" stroke-width="4"/>\`;
        svgContent += \`<path d="\${d}" fill="none" stroke="\${color}" stroke-width="2"/>\`;
      } else {
        svgContent += \`<path d="\${d}" fill="none" stroke="\${color}" stroke-width="2"/>\`;
      }
    }
  }

  // Control point handles (annotated mode only)
  if (showH && ann.rings.length > 0) {
    ann.rings.forEach((ring, ri) => {
      if (!ring.points) return;
      const color = RING_COLORS[ri] || '#FFFFFF';
      const labelFill = (ri === 6 || ri === 7) ? '#fff' : (ri >= 8 ? '#222' : '#111');
      ring.points.forEach((p, pi) => {
        svgContent += \`<circle class="handle" data-handle="ring_pt" data-ri="\${ri}" data-pi="\${pi}" cx="\${p[0].toFixed(1)}" cy="\${p[1].toFixed(1)}" r="4" fill="\${color}" stroke="#000" stroke-width="1" opacity="0.85"/>\`;
        if (pi === 0) {
          svgContent += \`<text x="\${p[0].toFixed(1)}" y="\${(p[1]+3.5).toFixed(1)}" text-anchor="middle" dominant-baseline="middle" fill="\${labelFill}" font-size="8" font-weight="bold" font-family="monospace" pointer-events="none">\${ri}</text>\`;
        }
      });
    });
  }

  // Boundary vertex handles (annotated mode only)
  if (showH && ann.paperBoundary) {
    ann.paperBoundary.forEach((p, i) => {
      svgContent += \`<circle class="handle" data-handle="boundary" data-idx="\${i}" cx="\${p[0].toFixed(1)}" cy="\${p[1].toFixed(1)}" r="10" fill="#00FF88" stroke="#000" stroke-width="2" opacity="0.9"/>\`;
      svgContent += \`<text x="\${p[0].toFixed(1)}" y="\${(p[1]-14).toFixed(1)}" text-anchor="middle" fill="#00FF88" font-size="11" font-family="monospace" pointer-events="none">\${i}</text>\`;
    });
  }

  // Detected arrows (generated mode only)
  if (showA && isGenerated && data.detected.arrows) {
    data.detected.arrows.forEach((arrow, ai) => {
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
      const { tip, nock, score } = arrow;
      const isMiss = score === 0;
      const isIncomplete = score === null || score === undefined;
      const shaftColor = isMiss ? '#888888' : '#FF8C00';
      const labelText = score === 0 ? 'M' : isIncomplete ? '?' : String(score);
      const mx = ((tip[0] + nock[0]) / 2).toFixed(1);
      const my = ((tip[1] + nock[1]) / 2 - 14).toFixed(1);
      svgContent += \`<line x1="\${tip[0].toFixed(1)}" y1="\${tip[1].toFixed(1)}" x2="\${nock[0].toFixed(1)}" y2="\${nock[1].toFixed(1)}" stroke="\${shaftColor}" stroke-width="2" stroke-dasharray="8 4"/>\`;
      const tipDash = isIncomplete ? 'stroke-dasharray="3 2"' : '';
      svgContent += \`<circle class="handle" data-handle="arrow_tip" data-ai="\${ai}" cx="\${tip[0].toFixed(1)}" cy="\${tip[1].toFixed(1)}" r="4" fill="#FF4500" stroke="#000" stroke-width="1" \${tipDash}/>\`;
      svgContent += \`<circle class="handle" data-handle="arrow_nock" data-ai="\${ai}" cx="\${nock[0].toFixed(1)}" cy="\${nock[1].toFixed(1)}" r="6" fill="#FFD700" stroke="#000" stroke-width="1"/>\`;
      svgContent += \`<text x="\${mx}" y="\${my}" text-anchor="middle" fill="white" font-size="10" font-weight="bold" font-family="monospace" pointer-events="none">\${labelText}</text>\`;
    });

    // Pending tip dot while placing nock
    if ((addArrowMode === 'place-nock' || addArrowMode === 'score-input') && pendingTip) {
      svgContent += \`<circle cx="\${pendingTip[0].toFixed(1)}" cy="\${pendingTip[1].toFixed(1)}" r="6" fill="#FF8C00" opacity="0.5" pointer-events="none"/>\`;
    }
  }

  const inAddMode = addArrowMode === 'place-tip' || addArrowMode === 'place-nock';
  const svgClass = inAddMode ? 'class="add-arrow-mode"' : '';
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

// ---- Drag & click handling ----
function attachSvgListeners() {
  const svg = document.getElementById('main-svg');
  if (!svg) return;

  svg.addEventListener('click', (e) => {
    if (viewMode === 'generated') return;
    if (e.ctrlKey || e.metaKey) {
      const ann = getAnnotation(currentIdx);
      if (!ann.paperBoundary) return;
      const mpt = svgPt(svg, e);
      const { edge, pt } = nearestBoundaryEdge(ann.paperBoundary, mpt.x, mpt.y);
      if (edge >= 0) {
        ann.paperBoundary.splice(edge + 1, 0, [Math.round(pt[0]), Math.round(pt[1])]);
        markModified(currentIdx);
        updateImageList();
        render();
      }
      return;
    }

    if (e.altKey) {
      const ann = getAnnotation(currentIdx);
      if (!ann.rings || ann.rings.length === 0) return;
      const mpt = svgPt(svg, e);
      const { ri, seg } = nearestRingSegment(ann.rings, mpt.x, mpt.y);
      if (ri >= 0) {
        ann.rings[ri].points.splice(seg + 1, 0, [mpt.x, mpt.y]);
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
      addArrowMode = 'place-nock';
      render();
    } else if (addArrowMode === 'place-nock') {
      const mpt = svgPt(svg, e);
      pendingNock = [mpt.x, mpt.y];
      addArrowMode = 'score-input';
      pickerContext = { type: 'new' };
      // AW-4: deduce score from rings
      const ann = getAnnotation(currentIdx);
      const rings = ann.rings && ann.rings.length ? ann.rings
        : (imageDataCache[IMAGES[currentIdx]]?.detected?.rings ?? []);
      const boundary = ann.paperBoundary ?? imageDataCache[IMAGES[currentIdx]]?.detected?.paperBoundary ?? null;
      const suggested = deduceScore(pendingTip[0], pendingTip[1], rings, boundary);
      showScorePicker('Score:', suggested === 0 ? 0 : suggested || undefined);
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
        if (!ann.paperBoundary || ann.paperBoundary.length <= 3) return;
        const idx = parseInt(el.getAttribute('data-idx'), 10);
        ann.paperBoundary.splice(idx, 1);
        markModified(currentIdx);
        updateImageList();
        render();
      } else if (handleType === 'ring_pt') {
        const ri = parseInt(el.getAttribute('data-ri'), 10);
        const pi = parseInt(el.getAttribute('data-pi'), 10);
        if (ann.rings[ri] && ann.rings[ri].points.length > 3) {
          ann.rings[ri].points.splice(pi, 1);
          markModified(currentIdx);
          updateImageList();
          render();
        }
      } else if (handleType === 'arrow_tip' || handleType === 'arrow_nock') {
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
      if (addArrowMode !== 'idle') return;
      e.preventDefault(); e.stopPropagation();
      const handleType = el.getAttribute('data-handle');

      if (handleType === 'ring_pt') {
        drag = { type: 'ring_pt', ri: parseInt(el.getAttribute('data-ri'), 10), pi: parseInt(el.getAttribute('data-pi'), 10) };
      } else if (handleType === 'boundary') {
        drag = { type: 'boundary', idx: parseInt(el.getAttribute('data-idx'), 10) };
      } else if (handleType === 'arrow_tip') {
        drag = { type: 'arrow_tip', ai: parseInt(el.getAttribute('data-ai'), 10) };
      } else if (handleType === 'arrow_nock') {
        drag = { type: 'arrow_nock', ai: parseInt(el.getAttribute('data-ai'), 10) };
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
          ann.rings[drag.ri].points[drag.pi] = [mpt.x, mpt.y];
        } else if (drag.type === 'boundary' && ann.paperBoundary) {
          ann.paperBoundary[drag.idx] = [Math.round(mpt.x), Math.round(mpt.y)];
        } else if (drag.type === 'arrow_tip') {
          ann.arrows[drag.ai].tip = [mpt.x, mpt.y];
        } else if (drag.type === 'arrow_nock') {
          ann.arrows[drag.ai].nock = [mpt.x, mpt.y];
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
  return ann && (
    (ann.arrows && ann.arrows.length > 0) ||
    (ann.rings  && ann.rings.length  > 0) ||
    ann.paperBoundary != null
  );
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
    // Modified dot (left)
    const modDot = isModified ? '<span class="mod-dot"></span>' : '<span class="spacer"></span>';
    // Generation status dot (right, AW-5)
    const dotClass = annotated ? 'annotated' : genState;
    const dotTitle = annotated ? 'has annotation' : genState === 'ready' ? 'detected' : genState === 'computing' ? 'computing…' : 'queued';
    const genDot = \`<span class="gen-dot \${dotClass}" title="\${dotTitle}"></span>\`;
    btn.innerHTML = modDot + '<span style="flex:1">' + filename + '</span>' + genDot;
    btn.addEventListener('click', () => selectImage(i));
    list.appendChild(btn);
  });
}

// ---- Data panel ----
function updateDataPanel() {
  const ann = getAnnotation(currentIdx);
  let html = '<table><thead><tr><th>Label</th><th>pts</th><th>cx</th><th>cy</th></tr></thead><tbody>';

  if (ann.paperBoundary) {
    html += \`<tr style="color:#00CC77"><td>Boundary</td><td>\${ann.paperBoundary.length} vtx</td><td>—</td><td>—</td></tr>\`;
  }

  ann.rings.forEach((ring, i) => {
    if (!ring.points) return;
    const cx = (ring.points.reduce((s, p) => s + p[0], 0) / ring.points.length).toFixed(0);
    const cy = (ring.points.reduce((s, p) => s + p[1], 0) / ring.points.length).toFixed(0);
    html += \`<tr><td>Ring \${i}</td><td>\${ring.points.length}</td><td>\${cx}</td><td>\${cy}</td></tr>\`;
  });

  html += '</tbody></table>';

  if (ann.arrows.length > 0) {
    html += \`<table><thead><tr><th>Arrow</th><th>tip</th><th>nock</th><th>score</th></tr></thead><tbody>\`;
    ann.arrows.forEach((arrow, ai) => {
      const s = arrow.score;
      const scoreLabel = s === 0 ? 'M' : (s === null || s === undefined) ? '?' : String(s);
      html += \`<tr>
        <td>A\${ai}</td>
        <td>(\${Math.round(arrow.tip[0])},\${Math.round(arrow.tip[1])})</td>
        <td>(\${Math.round(arrow.nock[0])},\${Math.round(arrow.nock[1])})</td>
        <td class="score-cell" data-ai="\${ai}">\${scoreLabel}</td>
      </tr>\`;
    });
    html += '</tbody></table>';
  } else {
    html += \`<div style="color:#555;font-size:0.72rem;margin-top:4px">No arrows annotated</div>\`;
  }

  document.getElementById('data-table').innerHTML = html;

  const saveBtn = document.getElementById('btn-save');
  const valid = isAnnotationValid(ann);
  saveBtn.classList.toggle('save-disabled', !valid);
  if (!valid) {
    const reasons = [];
    if (!ann.paperBoundary || ann.paperBoundary.length < 3)
      reasons.push(\`boundary=\${ann.paperBoundary ? ann.paperBoundary.length + ' pts (need ≥3)' : 'null'}\`);
    if (!ann.rings || ann.rings.length === 0) reasons.push('rings=0');
    if (!ann.arrows || ann.arrows.length === 0) reasons.push('arrows=0');
    console.log(\`[save-btn] disabled for \${IMAGES[currentIdx]}: \${reasons.join(', ')}\`);
  }
  saveBtn.title = valid ? '' : 'Cannot save: annotation needs a target border, rings, and at least one arrow';

  document.querySelectorAll('#data-table .score-cell').forEach(cell => {
    cell.addEventListener('click', () => {
      const ai = parseInt(cell.getAttribute('data-ai'), 10);
      pickerContext = { type: 'edit', ai };
      showScorePicker(\`Edit A\${ai} score:\`);
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

async function save() {
  console.log(\`[save] clicked — modified: [\${store.modified.join(', ') || 'none'}]\`);
  const ann = getAnnotation(currentIdx);
  if (!isAnnotationValid(ann)) {
    const missing = [];
    if (!ann.paperBoundary || ann.paperBoundary.length < 3) missing.push('boundary');
    if (!ann.rings || ann.rings.length === 0) missing.push('rings');
    if (!ann.arrows || ann.arrows.length === 0) missing.push('at least one arrow');
    const msg = 'Missing: ' + missing.join(', ');
    showSaveMsg(msg);
    console.log(\`[save] blocked — \${msg}\`);
    return;
  }
  const out = {};
  for (const filename of Object.keys(store.annotations)) {
    const ann = store.annotations[filename];
    out[filename] = { paperBoundary: ann.paperBoundary, rings: ann.rings, arrows: ann.arrows || [] };
  }
  const total = Object.keys(out).length;
  console.log(\`[save] sending \${total} annotation(s) (all known, server skips invalid ones)\`);
  for (const [filename, ann] of Object.entries(out)) {
    const modified = store.modified.includes(filename);
    const valid = isAnnotationValid(ann);
    console.log(\`[save]   \${modified ? '* ' : '  '}\${filename}: boundary=\${ann.paperBoundary ? ann.paperBoundary.length + ' pts' : 'null'} rings=\${ann.rings.length} arrows=\${ann.arrows.length} valid=\${valid}\${!valid ? ' ← will be skipped by server' : ''}\`);
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

// ---- Reset ----
function resetCurrent() {
  const filename = IMAGES[currentIdx];
  // Clear local state
  delete imageDataCache[filename];
  delete store.annotations[filename];
  store.modified = store.modified.filter(f => f !== filename);
  addArrowMode = 'idle';
  pendingTip = null;
  pendingNock = null;
  hideScorePicker();
  document.getElementById('btn-add-arrow').classList.remove('active-mode');
  updateImageList();
  renderLoading('Recomputing\u2026');
  // Ask server to re-run detection; SSE ready event will trigger selectImage reload
  fetch('/api/recompute/' + encodeURIComponent(filename), { method: 'POST' }).catch(() => {});
}

function resetAll() {
  if (!confirm('Reset all annotations?')) return;
  store = { annotations: {}, modified: [] };
  addArrowMode = 'idle';
  pendingTip = null;
  pendingNock = null;
  hideScorePicker();
  document.getElementById('btn-add-arrow').classList.remove('active-mode');
  updateImageList(); render();
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
    if (addArrowMode === 'score-input' && pickerContext) {
      if (pickerContext.type === 'new') {
        const ann = getAnnotation(currentIdx);
        ann.arrows.push({
          tip:  [Math.round(pendingTip[0]),  Math.round(pendingTip[1])],
          nock: [Math.round(pendingNock[0]), Math.round(pendingNock[1])],
          score: null,
        });
        pendingTip = null;
        pendingNock = null;
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
    } else if (addArrowMode === 'place-tip' || addArrowMode === 'place-nock') {
      pendingTip = null;
      pendingNock = null;
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
  }
});

// ---- Wire up ----
document.getElementById('btn-save').addEventListener('click', save);
document.getElementById('btn-reset').addEventListener('click', resetCurrent);
document.getElementById('btn-reset-all').addEventListener('click', resetAll);
document.getElementById('chk-rings').addEventListener('change', render);
document.getElementById('chk-boundary').addEventListener('change', render);
document.getElementById('chk-handles').addEventListener('change', render);
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
  } else if (addArrowMode === 'place-tip' || addArrowMode === 'place-nock') {
    pendingTip = null;
    pendingNock = null;
    addArrowMode = 'idle';
    document.getElementById('btn-add-arrow').classList.remove('active-mode');
    render();
  }
});

document.querySelectorAll('#score-picker button[data-score]').forEach(btn => {
  btn.addEventListener('click', () => {
    const raw = btn.getAttribute('data-score');
    onScoreSelect(raw === 'X' ? 'X' : parseInt(raw, 10));
  });
});
document.getElementById('score-skip').addEventListener('click', () => onScoreSelect(null));

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
    if (e.target === btn) return; // handled above
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
        // Update image list dot without full re-render
        const idx = IMAGES.indexOf(msg.filename);
        if (idx >= 0) updateImageList();
        // If user is looking at this image and it just became ready, reload it
        if (msg.state === 'ready' && idx === currentIdx && !imageDataCache[msg.filename]) {
          selectImage(idx);
        }
      }
    } catch {}
  };
  es.onerror = () => {}; // silently reconnect
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
            paperBoundary: ann.paperBoundary || null,
            rings: ann.rings || [],
            arrows: ann.arrows || [],
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
  // annotations: human-authored data only
  await db.query(`
    CREATE TABLE IF NOT EXISTS annotations (
      filename       TEXT PRIMARY KEY,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      arrows         JSONB NOT NULL DEFAULT '[]',
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  // generated: algorithm output, keyed by source hash
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

  // Wipe annotations rows with corrupt ring data (null coords from old buggy code)
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

  // generation status map for SSE (AW-5)
  type GenState = 'ready' | 'queued' | 'computing';
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

  // Server-side in-memory image cache
  const imageCache = new Map<string, CachedImageData>();

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

        const hashInGen = inGenerated.get(filename);
        let isReady = hashInGen === currentHash;
        console.log(`image request: ${filename}  hashInGen=${hashInGen ?? 'null'}  isReady=${isReady}`);

        let base64: string;
        let width: number;
        let height: number;
        let detectedRings: SplineRing[];
        let detectedBoundary: [number, number][] | null;
        let detectedArrows: { tip: [number, number]; nock: [number, number] | null }[];

        if (isReady) {
          // Fast path (AW-1): read from generated table
          console.log(`  fast path: loading ${filename}…`);
          ({ base64, width, height } = await loadImageBase64(imgPath));
          const { rows } = await db.query(
            'SELECT rings, paper_boundary, arrows FROM generated WHERE filename = $1',
            [filename],
          );
          detectedRings    = rows[0]?.rings          ?? [];
          detectedBoundary = rows[0]?.paper_boundary ?? null;
          detectedArrows   = rows[0]?.arrows         ?? [];

          // AW-2: validity check — fall back to slow path if data is corrupt
          if (!isValidDetected(detectedRings, detectedBoundary)) {
            logEvent('warn', 'invalid_generated', filename, 'null coordinates in stored data — recomputing');
            logEvent('info', 'fallback_slow_path', filename, 'invalid cache');
            await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
            inGenerated.delete(filename);
            generationStatus.set(filename, 'queued');
            isReady = false;
          }
        }

        if (!isReady) {
          // Slow path: run detection synchronously (blocks event loop ~45s per image)
          generationStatus.set(filename, 'computing');
          broadcastSSE({ type: 'status', filename, state: 'computing' });
          const entry = await processImage(imgPath);
          const ok = entry.result.success;
          console.log(`  ${filename} ... ${ok ? 'ok' : `FAILED: ${(entry.result as any).error}`}`);
          if (!ok) logEvent('error', 'detection_failed', filename, (entry.result as any).error ?? '');
          base64 = entry.base64;
          width = entry.width;
          height = entry.height;
          detectedRings    = ok ? entry.result.rings : [];
          detectedBoundary = ok && entry.result.paperBoundary ? entry.result.paperBoundary.points : null;
          detectedArrows   = entry.detectedArrows;

          // Write to generated table
          await db.query(
            `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (filename) DO UPDATE
               SET algorithm_hash = EXCLUDED.algorithm_hash,
                   paper_boundary = EXCLUDED.paper_boundary,
                   rings          = EXCLUDED.rings,
                   arrows         = EXCLUDED.arrows,
                   updated_at     = NOW()`,
            [filename, currentHash,
             JSON.stringify(detectedBoundary), JSON.stringify(detectedRings),
             JSON.stringify(detectedArrows)],
          );
          inGenerated.set(filename, currentHash);
          generationStatus.set(filename, 'ready');
          broadcastSSE({ type: 'status', filename, state: 'ready' });

          // Seed annotations if this image has no human annotation yet
          if (!inAnnotations.has(filename)) {
            await db.query(
              `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, '[]')
               ON CONFLICT (filename) DO NOTHING`,
              [filename, JSON.stringify(detectedBoundary), JSON.stringify(detectedRings)],
            );
            inAnnotations.add(filename);
          }
        }

        imageCache.set(filename, {
          base64,
          width,
          height,
          detected: { rings: detectedRings, paperBoundary: detectedBoundary, arrows: detectedArrows },
        });
      }

      respond(200, JSON.stringify(imageCache.get(filename)!));
      } catch (err) {
        console.error('Error processing image:', err);
        respond(500, JSON.stringify({ error: String(err) }));
      }

    } else if (req.method === 'GET' && req.url === '/api/annotations') {
      const { rows } = await db.query('SELECT filename, paper_boundary, rings, arrows FROM annotations');
      const out: Record<string, unknown> = {};
      for (const row of rows) {
        const ann = {
          paperBoundary: row.paper_boundary,
          rings: row.rings ?? [],
          arrows: row.arrows ?? [],
        };
        if (!isValidAnnotation(ann)) {
          await db.query('DELETE FROM annotations WHERE filename = $1', [row.filename]);
          inAnnotations.delete(row.filename);
          logEvent('warn', 'invalid_annotation_deleted', row.filename, 'deleted on load: incomplete annotation');
          continue;
        }
        out[row.filename] = ann;
      }
      respond(200, JSON.stringify(out));

    } else if (req.method === 'GET' && req.url?.startsWith('/api/annotation/')) {
      const filename = decodeURIComponent(req.url.slice('/api/annotation/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      const { rows } = await db.query(
        'SELECT paper_boundary, rings, arrows FROM annotations WHERE filename = $1',
        [filename],
      );
      if (rows.length === 0) { respond(404, '{"error":"not found"}'); return; }
      respond(200, JSON.stringify({
        paperBoundary: rows[0].paper_boundary,
        rings: rows[0].rings ?? [],
        arrows: rows[0].arrows ?? [],
      }));

    } else if (req.method === 'POST' && req.url?.startsWith('/api/recompute/')) {
      const filename = decodeURIComponent(req.url.slice('/api/recompute/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      // Clear caches so the next /api/image/ fetch re-runs detection
      imageCache.delete(filename);
      inGenerated.delete(filename);
      inAnnotations.delete(filename);
      await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
      await db.query('DELETE FROM annotations WHERE filename = $1', [filename]);
      generationStatus.set(filename, 'computing');
      broadcastSSE({ type: 'status', filename, state: 'computing' });
      respond(202, '{"status":"computing"}');
      // Run detection in background
      (async () => {
        try {
          const imgPath = path.join(IMAGES_DIR, filename);
          const entry = await processImage(imgPath);
          const ok = entry.result.success;
          if (!ok) logEvent('error', 'detection_failed', filename, (entry.result as any).error ?? '');
          const detectedRings    = ok ? entry.result.rings : [];
          const detectedBoundary = ok && entry.result.paperBoundary ? entry.result.paperBoundary.points : null;
          const detectedArrows   = entry.detectedArrows;
          await db.query(
            `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (filename) DO UPDATE
               SET algorithm_hash = EXCLUDED.algorithm_hash,
                   paper_boundary = EXCLUDED.paper_boundary,
                   rings          = EXCLUDED.rings,
                   arrows         = EXCLUDED.arrows,
                   updated_at     = NOW()`,
            [filename, currentHash,
             JSON.stringify(detectedBoundary), JSON.stringify(detectedRings),
             JSON.stringify(detectedArrows)],
          );
          await db.query(
            `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
             VALUES ($1, $2, $3, '[]')`,
            [filename, JSON.stringify(detectedBoundary), JSON.stringify(detectedRings)],
          );
          inGenerated.set(filename, currentHash);
          inAnnotations.add(filename);
          generationStatus.set(filename, 'ready');
          broadcastSSE({ type: 'status', filename, state: 'ready' });
        } catch (err) {
          console.error('Recompute error:', err);
          generationStatus.set(filename, 'queued');
          broadcastSSE({ type: 'status', filename, state: 'queued' });
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
          let saved = 0, skipped = 0;
          for (const [filename, ann] of Object.entries(data) as [string, any][]) {
            const normalized = {
              paperBoundary: ann.paperBoundary ?? null,
              rings: ann.rings ?? [],
              arrows: ann.arrows ?? [],
            };
            if (!isValidAnnotation(normalized)) {
              const reasons: string[] = [];
              if (!normalized.paperBoundary || normalized.paperBoundary.length < 3)
                reasons.push(`boundary=${normalized.paperBoundary ? normalized.paperBoundary.length + ' pts (need ≥3)' : 'null'}`);
              if (normalized.rings.length === 0) reasons.push('rings=0');
              if (normalized.arrows.length === 0) reasons.push('arrows=0');
              console.log(`[save]   SKIP ${filename}: ${reasons.join(', ')}`);
              logEvent('warn', 'save-skipped', filename, reasons.join(', '));
              skipped++;
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
              [
                filename,
                JSON.stringify(normalized.paperBoundary),
                JSON.stringify(normalized.rings),
                JSON.stringify(normalized.arrows),
              ],
            );
            console.log(`[save]   OK ${filename}: boundary=${normalized.paperBoundary!.length} pts, rings=${normalized.rings.length}, arrows=${normalized.arrows.length}`);
            logEvent('info', 'save-ok', filename, `rings=${normalized.rings.length} arrows=${normalized.arrows.length}`);
            saved++;
          }
          console.log(`[save] done: saved=${saved} skipped=${skipped}`);
          respond(200, JSON.stringify({ ok: true, saved, skipped }));
        } catch (e) {
          console.error('Save error:', e);
          logEvent('error', 'save-error', '', String(e));
          respond(500, `{"error":"${e}"}`);
        }
      });

    } else {
      respond(404, '');
    }
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Annotation tool: http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop.');
    if (!process.env.NO_BROWSER) require('child_process').exec(`open http://localhost:${PORT}`);

    // AW-5: start background detection queue after server is up
    const toProcess = filenames.filter(f => inGenerated.get(f) !== currentHash);
    if (toProcess.length === 0) return;
    console.log(`Background queue: ${toProcess.length} image(s) to process…`);

    (async () => {
      for (const filename of toProcess) {
        if (inGenerated.get(filename) === currentHash) continue; // already computed on-demand
        const imgPath = path.join(IMAGES_DIR, filename);
        generationStatus.set(filename, 'computing');
        broadcastSSE({ type: 'status', filename, state: 'computing' });
        try {
          const result = await runWorkerProcess(imgPath);
          if (!result.success) {
            logEvent('error', 'detection_failed', filename, result.error ?? '');
          }
          const { rings, paperBoundary, arrows } = result;
          await db.query(
            `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (filename) DO UPDATE
               SET algorithm_hash = EXCLUDED.algorithm_hash,
                   paper_boundary = EXCLUDED.paper_boundary,
                   rings          = EXCLUDED.rings,
                   arrows         = EXCLUDED.arrows,
                   updated_at     = NOW()`,
            [filename, currentHash,
             JSON.stringify(paperBoundary), JSON.stringify(rings), JSON.stringify(arrows)],
          );
          inGenerated.set(filename, currentHash);
          generationStatus.set(filename, 'ready');
          broadcastSSE({ type: 'status', filename, state: 'ready' });

          // Seed annotations if not yet present
          if (!inAnnotations.has(filename)) {
            await db.query(
              `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, '[]')
               ON CONFLICT (filename) DO NOTHING`,
              [filename, JSON.stringify(paperBoundary), JSON.stringify(rings)],
            );
            inAnnotations.add(filename);
          }
          console.log(`  [bg] ${filename} … ok`);
        } catch (err) {
          const msg = String(err);
          console.error(`  [bg] ${filename} … FAILED: ${msg}`);
          logEvent('error', 'detection_failed', filename, msg);
          generationStatus.set(filename, 'queued');
          broadcastSSE({ type: 'status', filename, state: 'queued' });
        }
      }
      console.log('Background queue complete.');
    })().catch(err => console.error('Background queue error:', err));
  });
}

main().catch(err => { console.error(err); process.exit(1); });
