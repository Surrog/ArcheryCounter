import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, EllipseData, ArcheryResult, Pixel } from '../src/targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const OUTPUT_PATH = path.resolve(__dirname, '../annotate.html');

const RING_COLORS = [
  '#FFD700', '#FFD700',
  '#E8000D', '#E8000D',
  '#006CB7', '#006CB7',
  '#888888', '#888888',
  '#FFFFFF', '#FFFFFF',
];

interface ImageEntry {
  filename: string;
  base64: string;
  width: number;
  height: number;
  result: ArcheryResult;
}

async function processImage(imgPath: string): Promise<ImageEntry> {
  const { Jimp } = require('jimp');
  const filename = path.basename(imgPath);

  const { rgba, width, height } = await loadImageNode(imgPath);

  const img = await Jimp.read(imgPath);
  img.scaleToFit({ w: 1200, h: 1200 });
  const base64 = await img.getBase64('image/jpeg');

  const result = findTarget(rgba, width, height);
  return { filename, base64, width, height, result };
}

function generateHtml(entries: ImageEntry[]): string {
  // Serialize image data as a JS constant embedded in the HTML
  const imagesData = entries.map(({ filename, base64, width, height, result }) => {
    const rings = result.success ? result.rings.map(r => ({
      centerX: r.centerX,
      centerY: r.centerY,
      width: r.width,
      height: r.height,
      angle: r.angle,
    })) : [];

    const paperBoundary = result.success && result.paperBoundary
      ? result.paperBoundary.map(p => [p.x, p.y])
      : null;

    return {
      filename,
      base64,
      width,
      height,
      detected: { rings, paperBoundary },
    };
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

    /* Sidebar */
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
    #controls label.file-btn { background: #444; color: #ccc; border-radius: 4px; padding: 6px 10px; cursor: pointer; font-size: 0.8rem; display: block; }
    #controls label.file-btn:hover { background: #555; }
    #controls input[type=file] { display: none; }

    /* Toolbar checkboxes */
    #toolbar { padding: 8px 14px; border-bottom: 1px solid #333; display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
    #toolbar label { font-size: 0.78rem; color: #aaa; display: flex; align-items: center; gap: 4px; cursor: pointer; }
    #legend { font-size: 0.72rem; color: #666; margin-left: auto; display: flex; gap: 10px; align-items: center; }
    .leg { display: flex; align-items: center; gap: 3px; }
    .leg svg { flex-shrink: 0; }

    /* Image list */
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

    /* Data panel */
    #data-panel { padding: 10px 14px; border-top: 1px solid #333; font-size: 0.72rem; overflow-y: auto; max-height: 200px; }
    #data-panel h3 { color: #888; margin-bottom: 6px; font-size: 0.75rem; font-weight: 600; }
    #data-panel table { width: 100%; border-collapse: collapse; }
    #data-panel th { color: #666; font-weight: 600; padding: 2px 4px; text-align: left; }
    #data-panel td { color: #aaa; font-family: monospace; padding: 2px 4px; border-top: 1px solid #2a2a2a; }

    /* Main canvas */
    #main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: auto; background: #111; padding: 16px; }
    #svg-container { position: relative; }
    .img-wrap { position: relative; display: inline-block; max-width: 100%; line-height: 0; }
    .img-wrap img { display: block; max-width: 100%; height: auto; user-select: none; -webkit-user-drag: none; }
    .img-wrap svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: visible; cursor: default; }
    .img-wrap svg.dragging { cursor: grabbing; }
    .handle { cursor: grab; }
    .handle:active { cursor: grabbing; }
  </style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Annotation Tool</h1>
    <div id="controls">
      <button id="btn-export">Export JSON</button>
      <label class="file-btn">Load JSON <input type="file" id="input-load" accept=".json"/></label>
      <button id="btn-reset">Reset image</button>
      <button id="btn-reset-all" class="danger">Reset all</button>
    </div>
  </div>
  <div id="toolbar">
    <label><input type="checkbox" id="chk-rings" checked/> Rings</label>
    <label><input type="checkbox" id="chk-boundary" checked/> Boundary</label>
    <label><input type="checkbox" id="chk-handles" checked/> Handles</label>
    <div id="legend">
      <span class="leg"><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="#FFD700" stroke="#000" stroke-width="1.5"/></svg> rx</span>
      <span class="leg"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="white" stroke="#FFD700" stroke-width="2"/></svg> ry</span>
      <span class="leg"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#4499FF" stroke="#fff" stroke-width="1.5"/></svg> rot</span>
      <span class="leg"><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="#FFD700" stroke="#000" stroke-width="1.5"/><circle cx="7" cy="7" r="3" fill="#000"/></svg> center</span>
      <span class="leg"><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="#00FF88" stroke="#000" stroke-width="1.5"/></svg> boundary</span>
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

// ---- Storage ----
const STORAGE_KEY = 'archery-ann';

function loadStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {}
  return { annotations: {}, modified: [] };
}

function saveStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch (e) {}
}

// ---- State ----
let store = loadStorage();
let currentIdx = 0;
let drag = null;

// ---- Overlay toggles ----
function showRings() { return document.getElementById('chk-rings').checked; }
function showBoundary() { return document.getElementById('chk-boundary').checked; }
function showHandles() { return document.getElementById('chk-handles').checked; }

// ---- Annotation helpers ----
function getDetected(idx) {
  const img = IMAGES[idx];
  return {
    paperBoundary: img.detected.paperBoundary
      ? img.detected.paperBoundary.map(p => [p[0], p[1]])
      : null,
    rings: img.detected.rings.map(r => ({
      centerX: r.centerX, centerY: r.centerY,
      width: r.width, height: r.height, angle: r.angle,
    })),
  };
}

function getAnnotation(idx) {
  const filename = IMAGES[idx].filename;
  if (!store.annotations[filename]) {
    store.annotations[filename] = getDetected(idx);
  }
  return store.annotations[filename];
}

function markModified(idx) {
  const filename = IMAGES[idx].filename;
  if (!store.modified.includes(filename)) {
    store.modified.push(filename);
  }
}

// ---- SVG coordinate helper ----
function svgPt(svg, e) {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

// ---- Render ----
function render() {
  const container = document.getElementById('svg-container');
  const img = IMAGES[currentIdx];
  const ann = getAnnotation(currentIdx);
  const W = img.width, H = img.height;
  const showR = showRings(), showB = showBoundary(), showH = showHandles();

  let svgContent = '';

  // Boundary
  if (ann.paperBoundary && showB) {
    const pts = ann.paperBoundary.map(p => \`\${p[0].toFixed(1)},\${p[1].toFixed(1)}\`).join(' ');
    svgContent += \`<polygon points="\${pts}" fill="none" stroke="#00FF88" stroke-width="3" stroke-dasharray="12 6" opacity="0.85"/>\`;
  }

  // Rings — draw outermost first
  if (showR && ann.rings.length > 0) {
    const ringsReversed = ann.rings.map((r, i) => ({ r, i })).reverse();
    for (const { r, i } of ringsReversed) {
      const rx = (r.width / 2).toFixed(1);
      const ry = (r.height / 2).toFixed(1);
      const cx = r.centerX.toFixed(1);
      const cy = r.centerY.toFixed(1);
      const angle = r.angle.toFixed(1);
      const color = RING_COLORS[i] || '#FFFFFF';
      const transform = \`rotate(\${angle},\${cx},\${cy})\`;
      if (i >= 8) {
        svgContent += \`<ellipse cx="\${cx}" cy="\${cy}" rx="\${rx}" ry="\${ry}" transform="\${transform}" fill="none" stroke="#000" stroke-width="4"/>\`;
        svgContent += \`<ellipse cx="\${cx}" cy="\${cy}" rx="\${rx}" ry="\${ry}" transform="\${transform}" fill="none" stroke="\${color}" stroke-width="2"/>\`;
      } else if (i >= 6) {
        svgContent += \`<ellipse cx="\${cx}" cy="\${cy}" rx="\${rx}" ry="\${ry}" transform="\${transform}" fill="none" stroke="#888" stroke-width="4"/>\`;
        svgContent += \`<ellipse cx="\${cx}" cy="\${cy}" rx="\${rx}" ry="\${ry}" transform="\${transform}" fill="none" stroke="\${color}" stroke-width="2"/>\`;
      } else {
        svgContent += \`<ellipse cx="\${cx}" cy="\${cy}" rx="\${rx}" ry="\${ry}" transform="\${transform}" fill="none" stroke="\${color}" stroke-width="2"/>\`;
      }
    }
  }

  // Center handle
  if (showH && ann.rings.length > 0) {
    const r0 = ann.rings[0];
    svgContent += \`<circle class="handle" data-handle="center" cx="\${r0.centerX.toFixed(1)}" cy="\${r0.centerY.toFixed(1)}" r="12" fill="#FFD700" stroke="#000" stroke-width="2" opacity="0.85"/>\`;
  }

  // Boundary corner handles
  if (showH && ann.paperBoundary) {
    const labels = ['TL','TR','BR','BL'];
    ann.paperBoundary.forEach((p, i) => {
      svgContent += \`<circle class="handle" data-handle="boundary" data-idx="\${i}" cx="\${p[0].toFixed(1)}" cy="\${p[1].toFixed(1)}" r="10" fill="#00FF88" stroke="#000" stroke-width="2" opacity="0.9"/>\`;
      svgContent += \`<text x="\${p[0].toFixed(1)}" y="\${(p[1] - 14).toFixed(1)}" text-anchor="middle" fill="#00FF88" font-size="11" font-family="monospace" pointer-events="none">\${labels[i]}</text>\`;
    });
  }

  // Ring handles: rx (major axis), ry (minor axis), rot (rotation) — 3 per ring
  if (showH && ann.rings.length > 0) {
    ann.rings.forEach((r, i) => {
      const a = r.angle * Math.PI / 180;
      const rx = r.width / 2;
      const ry = r.height / 2;
      const color = RING_COLORS[i] || '#FFFFFF';
      const labelFill = (i === 6 || i === 7) ? '#fff' : '#111';

      // rx handle — tip of major axis, filled with ring colour, labelled with ring index
      const rxHx = r.centerX + rx * Math.cos(a);
      const rxHy = r.centerY + rx * Math.sin(a);
      svgContent += \`<circle class="handle" data-handle="rx" data-idx="\${i}" cx="\${rxHx.toFixed(1)}" cy="\${rxHy.toFixed(1)}" r="8" fill="\${color}" stroke="#000" stroke-width="2" opacity="0.92"/>\`;
      svgContent += \`<text x="\${rxHx.toFixed(1)}" y="\${(rxHy + 3.5).toFixed(1)}" text-anchor="middle" dominant-baseline="middle" fill="\${labelFill}" font-size="9" font-weight="bold" font-family="monospace" pointer-events="none">\${i}</text>\`;

      // ry handle — tip of minor axis, white fill / ring-colour stroke
      const ryHx = r.centerX - ry * Math.sin(a);
      const ryHy = r.centerY + ry * Math.cos(a);
      svgContent += \`<circle class="handle" data-handle="ry" data-idx="\${i}" cx="\${ryHx.toFixed(1)}" cy="\${ryHy.toFixed(1)}" r="7" fill="white" stroke="\${color}" stroke-width="2.5" opacity="0.92"/>\`;

      // rotation handle — beyond rx tip, blue
      const rotHx = r.centerX + (rx + 22) * Math.cos(a);
      const rotHy = r.centerY + (rx + 22) * Math.sin(a);
      svgContent += \`<circle class="handle" data-handle="rot" data-idx="\${i}" cx="\${rotHx.toFixed(1)}" cy="\${rotHy.toFixed(1)}" r="6" fill="#4499FF" stroke="#fff" stroke-width="1.5" opacity="0.92"/>\`;
    });
  }

  const wrapEl = \`<div class="img-wrap">
  <img src="\${img.base64}" alt="\${img.filename}" draggable="false"/>
  <svg id="main-svg" viewBox="0 0 \${W} \${H}" xmlns="http://www.w3.org/2000/svg">
    \${svgContent}
  </svg>
</div>\`;

  container.innerHTML = wrapEl;
  attachSvgListeners();
  updateDataPanel();
}

// ---- Drag handling ----
function attachSvgListeners() {
  const svg = document.getElementById('main-svg');
  if (!svg) return;

  svg.querySelectorAll('.handle').forEach(el => {
    el.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const ann = getAnnotation(currentIdx);
      const pt = svgPt(svg, e);
      const handleType = el.getAttribute('data-handle');
      const idx = parseInt(el.getAttribute('data-idx') || '0', 10);

      if (handleType === 'center') {
        drag = { type: 'center', idx: 0, ox: pt.x, oy: pt.y };
      } else if (handleType === 'boundary') {
        drag = { type: 'boundary', idx, ox: pt.x, oy: pt.y };
      } else if (handleType === 'rx') {
        drag = { type: 'rx', idx };
      } else if (handleType === 'ry') {
        drag = { type: 'ry', idx };
      } else if (handleType === 'rot') {
        drag = { type: 'rot', idx };
      }

      const svgEl0 = document.getElementById('main-svg');
      if (svgEl0) svgEl0.classList.add('dragging');

      const onMove = (me) => {
        if (!drag) return;
        const liveSvg = document.getElementById('main-svg');
        if (!liveSvg) return;
        const mpt = svgPt(liveSvg, me);
        const ann = getAnnotation(currentIdx);

        if (drag.type === 'center') {
          const dx = mpt.x - drag.ox;
          const dy = mpt.y - drag.oy;
          for (const ring of ann.rings) {
            ring.centerX += dx;
            ring.centerY += dy;
          }
          drag.ox = mpt.x;
          drag.oy = mpt.y;
        } else if (drag.type === 'boundary' && ann.paperBoundary) {
          ann.paperBoundary[drag.idx] = [Math.round(mpt.x), Math.round(mpt.y)];
        } else if (drag.type === 'rx') {
          const ring = ann.rings[drag.idx];
          const dist = Math.hypot(mpt.x - ring.centerX, mpt.y - ring.centerY);
          ring.width = Math.max(4, 2 * dist);
        } else if (drag.type === 'ry') {
          const ring = ann.rings[drag.idx];
          const dist = Math.hypot(mpt.x - ring.centerX, mpt.y - ring.centerY);
          ring.height = Math.max(2, 2 * dist);
        } else if (drag.type === 'rot') {
          const ring = ann.rings[drag.idx];
          ring.angle = Math.atan2(mpt.y - ring.centerY, mpt.x - ring.centerX) * 180 / Math.PI;
        }

        render();
      };

      const onUp = () => {
        drag = null;
        const svgElUp = document.getElementById('main-svg');
        if (svgElUp) svgElUp.classList.remove('dragging');
        markModified(currentIdx);
        saveStorage();
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
    btn.addEventListener('click', () => {
      currentIdx = i;
      updateImageList();
      render();
    });
    list.appendChild(btn);
  });
}

// ---- Data panel ----
function updateDataPanel() {
  const ann = getAnnotation(currentIdx);
  let html = '<table><thead><tr><th>Label</th><th>cx/x</th><th>cy/y</th><th>w</th><th>h</th></tr></thead><tbody>';

  if (ann.paperBoundary) {
    const labels = ['TL','TR','BR','BL'];
    ann.paperBoundary.forEach((p, i) => {
      html += \`<tr style="color:#00CC77"><td>Bnd \${labels[i]}</td><td>\${p[0].toFixed(0)}</td><td>\${p[1].toFixed(0)}</td><td>—</td><td>—</td></tr>\`;
    });
  }

  ann.rings.forEach((r, i) => {
    html += \`<tr><td>Ring \${i}</td><td>\${r.centerX.toFixed(1)}</td><td>\${r.centerY.toFixed(1)}</td><td>\${r.width.toFixed(1)}</td><td>\${r.height.toFixed(1)}</td></tr>\`;
  });

  html += '</tbody></table>';
  document.getElementById('data-table').innerHTML = html;
}

// ---- Export ----
function exportJson() {
  const out = {};
  for (const filename of Object.keys(store.annotations)) {
    const ann = store.annotations[filename];
    out[filename] = {
      paperBoundary: ann.paperBoundary,
      rings: ann.rings.map(r => ({
        centerX: r.centerX,
        centerY: r.centerY,
        width: r.width,
        height: r.height,
        angle: r.angle,
      })),
    };
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'annotations.json';
  a.click();
  URL.revokeObjectURL(url);
}

// ---- Load JSON ----
function loadJson(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      for (const [filename, ann] of Object.entries(data)) {
        store.annotations[filename] = ann;
        if (!store.modified.includes(filename)) store.modified.push(filename);
      }
      saveStorage();
      updateImageList();
      render();
    } catch (err) {
      alert('Failed to parse JSON: ' + err);
    }
  };
  reader.readAsText(file);
}

// ---- Reset ----
function resetCurrent() {
  const filename = IMAGES[currentIdx].filename;
  store.annotations[filename] = getDetected(currentIdx);
  store.modified = store.modified.filter(f => f !== filename);
  saveStorage();
  updateImageList();
  render();
}

function resetAll() {
  if (!confirm('Reset all annotations? This will clear localStorage.')) return;
  store = { annotations: {}, modified: [] };
  saveStorage();
  updateImageList();
  render();
}

// ---- Wire up controls ----
document.getElementById('btn-export').addEventListener('click', exportJson);
document.getElementById('btn-reset').addEventListener('click', resetCurrent);
document.getElementById('btn-reset-all').addEventListener('click', resetAll);

document.getElementById('input-load').addEventListener('change', (e) => {
  if (e.target.files.length > 0) loadJson(e.target.files[0]);
});

document.getElementById('chk-rings').addEventListener('change', render);
document.getElementById('chk-boundary').addEventListener('change', render);
document.getElementById('chk-handles').addEventListener('change', render);

// ---- Init ----
updateImageList();
render();
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

  console.log(`Processing ${jpgFiles.length} image(s) from ${IMAGES_DIR}\n`);

  const entries: ImageEntry[] = [];

  for (const imgPath of jpgFiles) {
    const filename = path.basename(imgPath);
    process.stdout.write(`  ${filename} ... `);
    try {
      const entry = await processImage(imgPath);
      entries.push(entry);
      const label = entry.result.success ? 'ok' : `FAILED: ${entry.result.error}`;
      console.log(label);
    } catch (err) {
      console.log(`EXCEPTION: ${err}`);
      entries.push({
        filename,
        base64: '',
        width: 0,
        height: 0,
        result: { success: false, rings: [], error: String(err) },
      });
    }
  }

  const html = generateHtml(entries);
  fs.writeFileSync(OUTPUT_PATH, html, 'utf8');

  const passCount = entries.filter(e => e.result.success).length;
  console.log(`\nWrote: ${OUTPUT_PATH}`);
  console.log(`Result: ${passCount}/${entries.length} passed`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
