import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import { Pool } from 'pg';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, ArcheryResult } from '../src/targetDetection';
import { ellipseToSplinePoints } from '../src/spline';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const ANNOTATIONS_PATH = path.resolve(__dirname, '../images/annotate.json');
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

interface SplineRing { points: [number, number][]; }

interface ImageEntry {
  filename: string;
  base64: string;
  width: number;
  height: number;
  result: ArcheryResult;
}

async function loadImageBase64(imgPath: string): Promise<{ base64: string; width: number; height: number }> {
  const { Jimp } = require('jimp');
  const img = await Jimp.read(imgPath);
  img.scaleToFit({ w: 1200, h: 1200 });
  const base64 = await img.getBase64('image/jpeg');
  return { base64, width: img.width, height: img.height };
}

async function processImage(imgPath: string): Promise<ImageEntry> {
  const filename = path.basename(imgPath);
  const { rgba, width, height } = await loadImageNode(imgPath);
  const { base64 } = await loadImageBase64(imgPath);
  const result = findTarget(rgba, width, height);
  return { filename, base64, width, height, result };
}

function generateHtml(entries: ImageEntry[]): string {
  const imagesData = entries.map(({ filename, base64, width, height, result }) => {
    const rings: SplineRing[] = result.success
      ? result.rings.map((r: any) => ({
          points: ellipseToSplinePoints(r.centerX, r.centerY, r.width / 2, r.height / 2, r.angle, K_POINTS),
        }))
      : [];

    const paperBoundary = result.success && result.paperBoundary
      ? result.paperBoundary.points
      : null;

    return { filename, base64, width, height, detected: { rings, paperBoundary } };
  });

  const imagesJson = JSON.stringify(imagesData);

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

    #toolbar { padding: 8px 14px; border-bottom: 1px solid #333; display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
    #toolbar label { font-size: 0.78rem; color: #aaa; display: flex; align-items: center; gap: 4px; cursor: pointer; }
    #legend { font-size: 0.7rem; color: #666; width: 100%; margin-top: 2px; }

    #score-picker { display: none; gap: 4px; align-items: center; flex-wrap: wrap; width: 100%; padding-top: 6px; }
    #score-picker .sp-label { font-size: 0.78rem; color: #FF8C00; white-space: nowrap; margin-right: 2px; }
    #score-picker button { padding: 3px 7px; font-size: 0.78rem; border: 1px solid #555; border-radius: 3px; background: #2a2a2a; color: #ddd; cursor: pointer; }
    #score-picker button:hover { background: #3a3a3a; }
    #score-picker button.gold { background: #5a4a00; color: #FFD700; border-color: #FFD700; font-weight: bold; }
    #score-picker button.gold:hover { background: #7a6400; }
    #score-picker button.miss { background: #3a2a2a; color: #999; }
    #score-picker button.miss:hover { background: #4a3a3a; }

    #img-list { flex: 1; overflow-y: auto; padding: 8px 0; }
    .img-btn {
      width: 100%; text-align: left; background: none; border: none; color: #ccc;
      padding: 7px 14px; font-size: 0.78rem; cursor: pointer; display: flex; align-items: center; gap: 6px;
      border-left: 3px solid transparent;
    }
    .img-btn:hover { background: #2a2a2a; }
    .img-btn.active { background: #1a3a5c; border-left-color: #4a9eff; color: #fff; }
    .img-btn.modified { color: #f0a030; }
    .img-btn .dot { width: 6px; height: 6px; border-radius: 50%; background: #f0a030; flex-shrink: 0; }
    .img-btn .spacer { width: 6px; flex-shrink: 0; }

    #data-panel { padding: 10px 14px; border-top: 1px solid #333; font-size: 0.72rem; overflow-y: auto; max-height: 240px; }
    #data-panel h3 { color: #888; margin-bottom: 6px; font-size: 0.75rem; font-weight: 600; }
    #data-panel table { width: 100%; border-collapse: collapse; margin-bottom: 6px; }
    #data-panel th { color: #666; font-weight: 600; padding: 2px 4px; text-align: left; }
    #data-panel td { color: #aaa; font-family: monospace; padding: 2px 4px; border-top: 1px solid #2a2a2a; }
    #data-panel td.score-cell { cursor: pointer; color: #FF8C00; }
    #data-panel td.score-cell:hover { text-decoration: underline; }

    #main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: auto; background: #111; padding: 16px; }
    #svg-container { position: relative; }
    .img-wrap { position: relative; display: inline-block; max-width: 100%; line-height: 0; }
    .img-wrap img { display: block; max-width: 100%; height: auto; user-select: none; -webkit-user-drag: none; }
    .img-wrap svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: visible; cursor: default; }
    .img-wrap svg.dragging { cursor: grabbing; }
    .img-wrap svg.add-arrow-mode { cursor: crosshair; }
    .handle { cursor: grab; }
    .handle:active { cursor: grabbing; }
  </style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Annotation Tool</h1>
    <div id="controls">
      <button id="btn-save">Save</button>
      <button id="btn-reset">Reset image</button>
      <button id="btn-add-arrow">Add arrow (A)</button>
      <button id="btn-reset-all" class="danger">Reset all</button>
    </div>
  </div>
  <div id="toolbar">
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
      Ctrl+click boundary: add vertex · Shift+click vertex/arrow: remove · A: add arrow
    </div>
  </div>
  <div id="img-list"></div>
  <div id="data-panel">
    <h3>Current annotation</h3>
    <div id="data-table"></div>
  </div>
</div>

<div id="main">
  <div id="svg-container"></div>
</div>

<script>
const IMAGES = ${imagesJson};
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

// ---- State ----
let store = { annotations: {}, modified: [] };
let currentIdx = 0;
let drag = null;
let addArrowMode = 'idle'; // 'idle' | 'place-tip' | 'place-nock' | 'score-input'
let pendingTip = null;
let pendingNock = null;
let pickerContext = null; // { type: 'new' } | { type: 'edit', ai: number } | null

// ---- Overlay toggles ----
function showRings()    { return document.getElementById('chk-rings').checked; }
function showBoundary() { return document.getElementById('chk-boundary').checked; }
function showHandles()  { return document.getElementById('chk-handles').checked; }
function showArrows()   { return document.getElementById('chk-arrows').checked; }

// ---- Annotation helpers ----
function getDetected(idx) {
  const img = IMAGES[idx];
  return {
    paperBoundary: img.detected.paperBoundary ? img.detected.paperBoundary.map(p => [p[0], p[1]]) : null,
    rings: img.detected.rings,
  };
}

function getAnnotation(idx) {
  const filename = IMAGES[idx].filename;
  if (!store.annotations[filename]) store.annotations[filename] = { ...getDetected(idx), arrows: [] };
  const ann = store.annotations[filename];
  if (!ann.arrows) ann.arrows = [];
  return ann;
}

function markModified(idx) {
  const filename = IMAGES[idx].filename;
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

// ---- Score picker ----
function showScorePicker(label) {
  const picker = document.getElementById('score-picker');
  picker.style.display = 'flex';
  document.getElementById('score-prompt').textContent = label || 'Score:';
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

// ---- Render ----
function render() {
  const container = document.getElementById('svg-container');
  const img = IMAGES[currentIdx];
  const ann = getAnnotation(currentIdx);
  const W = img.width, H = img.height;
  const showR = showRings(), showB = showBoundary(), showH = showHandles(), showA = showArrows();

  let svgContent = '';

  // Boundary polygon
  if (ann.paperBoundary && showB) {
    const pts = ann.paperBoundary.map(p => \`\${p[0].toFixed(1)},\${p[1].toFixed(1)}\`).join(' ');
    svgContent += \`<polygon points="\${pts}" fill="none" stroke="#00FF88" stroke-width="3" stroke-dasharray="12 6" opacity="0.85"/>\`;
  }

  // Ring splines — draw outermost first
  if (showR && ann.rings.length > 0) {
    for (let i = ann.rings.length - 1; i >= 0; i--) {
      const ring = ann.rings[i];
      if (!ring.points || ring.points.length < 3) continue;
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

  // Control point handles
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

  // Boundary vertex handles
  if (showH && ann.paperBoundary) {
    ann.paperBoundary.forEach((p, i) => {
      svgContent += \`<circle class="handle" data-handle="boundary" data-idx="\${i}" cx="\${p[0].toFixed(1)}" cy="\${p[1].toFixed(1)}" r="10" fill="#00FF88" stroke="#000" stroke-width="2" opacity="0.9"/>\`;
      svgContent += \`<text x="\${p[0].toFixed(1)}" y="\${(p[1]-14).toFixed(1)}" text-anchor="middle" fill="#00FF88" font-size="11" font-family="monospace" pointer-events="none">\${i}</text>\`;
    });
  }

  // Arrows
  if (showA) {
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
  <img src="\${img.base64}" alt="\${img.filename}" draggable="false"/>
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
      showScorePicker('Score:');
      render();
    }
  });

  svg.querySelectorAll('.handle').forEach(el => {
    el.addEventListener('click', (e) => {
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
      } else if (handleType === 'arrow_tip' || handleType === 'arrow_nock') {
        const ai = parseInt(el.getAttribute('data-ai'), 10);
        ann.arrows.splice(ai, 1);
        markModified(currentIdx);
        updateImageList();
        render();
      }
    });

    el.addEventListener('mousedown', (e) => {
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
function updateImageList() {
  const list = document.getElementById('img-list');
  list.innerHTML = '';
  IMAGES.forEach((img, i) => {
    const isModified = store.modified.includes(img.filename);
    const isActive = i === currentIdx;
    const btn = document.createElement('button');
    btn.className = 'img-btn' + (isActive ? ' active' : '') + (isModified ? ' modified' : '');
    const dot = isModified ? '<span class="dot"></span>' : '<span class="spacer"></span>';
    btn.innerHTML = dot + '<span>' + img.filename + '</span>';
    btn.addEventListener('click', () => { currentIdx = i; updateImageList(); render(); });
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

  // Arrows section
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

  document.querySelectorAll('#data-table .score-cell').forEach(cell => {
    cell.addEventListener('click', () => {
      const ai = parseInt(cell.getAttribute('data-ai'), 10);
      pickerContext = { type: 'edit', ai };
      showScorePicker(\`Edit A\${ai} score:\`);
    });
  });
}

// ---- Save to DB ----
async function save() {
  const out = {};
  for (const filename of Object.keys(store.annotations)) {
    const ann = store.annotations[filename];
    out[filename] = { paperBoundary: ann.paperBoundary, rings: ann.rings, arrows: ann.arrows || [] };
  }
  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(out),
    });
    if (!res.ok) throw new Error(\`HTTP \${res.status}\`);
  } catch (e) {
    alert('Save failed: ' + e);
    return;
  }
  store.modified = [];
  updateImageList();
}

// ---- Reset ----
function resetCurrent() {
  const filename = IMAGES[currentIdx].filename;
  store.annotations[filename] = { ...getDetected(currentIdx), arrows: [] };
  store.modified = store.modified.filter(f => f !== filename);
  addArrowMode = 'idle';
  pendingTip = null;
  pendingNock = null;
  hideScorePicker();
  document.getElementById('btn-add-arrow').classList.remove('active-mode');
  updateImageList(); render();
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

  if (e.key === 'Escape') {
    if (addArrowMode === 'score-input' && pickerContext) {
      if (pickerContext.type === 'new') {
        // Commit with null score
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

  if ((e.key === 'a' || e.key === 'A') && addArrowMode !== 'score-input') {
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

document.getElementById('btn-add-arrow').addEventListener('click', () => {
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

// ---- Init ----
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
    updateImageList();
    render();
  })
  .catch(() => { updateImageList(); render(); });
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

  // --- DB setup ---
  await db.query(`
    CREATE TABLE IF NOT EXISTS annotations (
      filename       TEXT PRIMARY KEY,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  // P8-T1: add arrows column if not present
  await db.query(`ALTER TABLE annotations ADD COLUMN IF NOT EXISTS arrows JSONB NOT NULL DEFAULT '[]'`);
  console.log('Table ready.');

  const { rows: existing } = await db.query('SELECT filename FROM annotations');
  const inDb = new Set(existing.map((r: any) => r.filename));

  const newImages = jpgFiles.filter(p => !inDb.has(path.basename(p)));
  console.log(`Images: ${jpgFiles.length} total, ${inDb.size} in DB, ${newImages.length} to process\n`);

  const entries: ImageEntry[] = [];

  for (const imgPath of jpgFiles) {
    const filename = path.basename(imgPath);
    process.stdout.write(`  ${filename} ... `);
    try {
      let entry: ImageEntry;
      if (inDb.has(filename)) {
        const { base64, width, height } = await loadImageBase64(imgPath);
        entry = { filename, base64, width, height, result: { success: true, rings: [], paperBoundary: undefined } };
        console.log('cached');
      } else {
        entry = await processImage(imgPath);
        console.log(entry.result.success ? 'ok' : `FAILED: ${entry.result.error}`);
      }
      entries.push(entry);
    } catch (err) {
      console.log(`EXCEPTION: ${err}`);
      entries.push({ filename, base64: '', width: 0, height: 0, result: { success: false, rings: [], error: String(err) } });
    }
  }

  const passCount = entries.filter(e => !inDb.has(e.filename) && e.result.success).length;
  const processedCount = newImages.length;
  if (processedCount > 0) console.log(`\nResult: ${passCount}/${processedCount} new images passed`);

  const savedAnnotations = fs.existsSync(ANNOTATIONS_PATH)
    ? JSON.parse(fs.readFileSync(ANNOTATIONS_PATH, 'utf8'))
    : {};

  let seeded = 0;
  for (const entry of entries.filter(e => !inDb.has(e.filename))) {
    const fromFile = savedAnnotations[entry.filename];
    const paperBoundary = fromFile?.paperBoundary
      ?? (entry.result.success && entry.result.paperBoundary ? entry.result.paperBoundary.points : null);
    const rings = fromFile?.rings
      ?? (entry.result.success ? entry.result.rings.map((r: any) => ({
          points: ellipseToSplinePoints(r.centerX, r.centerY, r.width / 2, r.height / 2, r.angle, K_POINTS),
        })) : []);

    await db.query(
      `INSERT INTO annotations (filename, paper_boundary, rings, arrows) VALUES ($1, $2, $3, $4)`,
      [entry.filename, JSON.stringify(paperBoundary), JSON.stringify(rings), JSON.stringify([])],
    );
    seeded++;
  }
  if (seeded > 0) console.log(`Seeded ${seeded} new image(s).`);
  else console.log('All images already in DB, no seed needed.');

  const html = generateHtml(entries);

  // --- HTTP server ---
  const server = http.createServer(async (req, res) => {
    const respond = (status: number, body: string, type = 'application/json') => {
      res.writeHead(status, { 'Content-Type': type });
      res.end(body);
    };

    if (req.method === 'GET' && req.url === '/') {
      respond(200, html, 'text/html; charset=utf-8');

    } else if (req.method === 'GET' && req.url === '/api/annotations') {
      const { rows } = await db.query('SELECT filename, paper_boundary, rings, arrows FROM annotations');
      const out: Record<string, unknown> = {};
      for (const row of rows) {
        out[row.filename] = { paperBoundary: row.paper_boundary, rings: row.rings, arrows: row.arrows };
      }
      respond(200, JSON.stringify(out));

    } else if (req.method === 'POST' && req.url === '/api/save') {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      req.on('end', async () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          for (const [filename, ann] of Object.entries(data) as [string, any][]) {
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
                JSON.stringify(ann.paperBoundary ?? null),
                JSON.stringify(ann.rings ?? []),
                JSON.stringify(ann.arrows ?? []),
              ],
            );
          }
          console.log(`Saved ${Object.keys(data).length} annotations`);
          respond(200, '{"ok":true}');
        } catch (e) {
          console.error('Save error:', e);
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
  });
}

main().catch(err => { console.error(err); process.exit(1); });
