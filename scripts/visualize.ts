import * as path from 'path';
import * as fs from 'fs';
import * as crypto from 'crypto';
import { Jimp } from 'jimp';
import { Pool } from 'pg';
import { findTarget, ArcheryResult, TargetBoundary, ColourCalibration, Pixel, RayDebugEntry } from '../src/targetDetection';
import { detectArrowsNN } from '../src/arrowDetector';
import type { ScoredArrow } from '../src/scoring';
import { SplineRing, sampleClosedSpline } from '../src/spline';

const IMAGES_DIR  = path.resolve(__dirname, '../images');
const OUTPUT_PATH = path.resolve(__dirname, '../report.html');

const args       = process.argv.slice(2);
const modelIdx   = args.indexOf('--model');
const MODEL_PATH = modelIdx >= 0
  ? path.resolve(args[modelIdx + 1])
  : path.resolve(__dirname, 'nn/arrow_detector_fp32.onnx');

// ── algorithm cache (mirrors regen-generated.ts) ──────────────────────────────

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

function computeAlgorithmHash(): string {
  const hash = crypto.createHash('sha256');
  // Hash source files
  for (const f of [
    path.resolve(__dirname, '../src/targetDetection.ts'),
    path.resolve(__dirname, '../src/arrowDetector.ts'),
  ]) {
    if (fs.existsSync(f)) hash.update(fs.readFileSync(f));
  }
  // Include model file size+mtime as a lightweight proxy for model version
  if (fs.existsSync(MODEL_PATH)) {
    const st = fs.statSync(MODEL_PATH);
    hash.update(`${MODEL_PATH}:${st.size}:${st.mtimeMs}`);
  }
  return hash.digest('hex').slice(0, 16);
}

function scalePoints(points: [number, number][], sx: number, sy: number): [number, number][] {
  return points.map(([x, y]) => [x * sx, y * sy]);
}

function scaleRings(rings: { points: [number, number][] }[], sx: number, sy: number): SplineRing[] {
  return rings.map(r => ({ points: scalePoints(r.points, sx, sy) as [number, number][] }));
}

function scaleBoundary(pb: any, sx: number, sy: number): TargetBoundary | undefined {
  if (!pb) return undefined;
  const pts: [number, number][] = Array.isArray(pb) ? pb : pb.points;
  return { points: scalePoints(pts, sx, sy) } as unknown as TargetBoundary;
}

function scaleArrows(arrows: any[], sx: number, sy: number): ScoredArrow[] {
  return arrows.map(a => ({
    tip:   [a.tip[0] * sx, a.tip[1] * sy] as [number, number],
    score: a.score ?? 0,
  }));
}

// Archery standard ring colours — index 0 (bullseye) to index 9 (outermost)
const RING_COLORS = [
  '#FFD700', '#FFD700', // 0,1  gold  — score 10, 9
  '#E8000D', '#E8000D', // 2,3  red   — score  8, 7
  '#006CB7', '#006CB7', // 4,5  blue  — score  6, 5
  '#444444', '#444444', // 6,7  black — score  4, 3
  '#FFFFFF', '#FFFFFF', // 8,9  white — score  2, 1
];

interface ImageEntry {
  filename: string;
  base64: string;
  width: number;
  height: number;
  result: ArcheryResult;
  arrows: ScoredArrow[];
}

async function processImage(imgPath: string, currentHash: string): Promise<ImageEntry> {
  const filename = path.basename(imgPath);

  // Scale to ≤1200px — matches loadImageNode() used by detect-worker/globalSetup,
  // so cached DB coordinates are already in the same space as the display image.
  const img = await Jimp.read(imgPath);
  img.scaleToFit({ w: 1200, h: 1200 });
  const { width, height } = img.bitmap;
  const base64 = await img.getBase64('image/jpeg');

  // ── cache check ──────────────────────────────────────────────────────────
  const cached = await db.query(
    `SELECT algorithm_hash, paper_boundary, rings, arrows, width, height
     FROM generated WHERE filename = $1`,
    [filename],
  );

  if (cached.rows.length > 0 && cached.rows[0].algorithm_hash === currentHash) {
    const row = cached.rows[0];
    // Stored coords are in (row.width × row.height) space; scale to current display size.
    const sx = width  / (row.width  ?? width);
    const sy = height / (row.height ?? height);
    const rings    = scaleRings(row.rings ?? [], sx, sy);
    const boundary = scaleBoundary(row.paper_boundary, sx, sy);
    const arrows   = scaleArrows(row.arrows ?? [], sx, sy);
    const result: ArcheryResult = { success: rings.length > 0, rings, paperBoundary: boundary };
    return { filename, base64, width, height, result, arrows };
  }

  // ── cache miss: run detection on the same ≤1200px image ──────────────────
  const rgba = new Uint8Array(img.bitmap.data.buffer);
  const result = findTarget(rgba, width, height);
  const arrowsFull = result.success
    ? await detectArrowsNN(rgba, width, height, MODEL_PATH)
    : [];

  // Persist to DB
  await db.query(
    `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
     VALUES ($1, $2, $3, $4, $5, $6, $7)
     ON CONFLICT (filename) DO UPDATE SET
       algorithm_hash = EXCLUDED.algorithm_hash,
       paper_boundary = EXCLUDED.paper_boundary,
       rings          = EXCLUDED.rings,
       arrows         = EXCLUDED.arrows,
       width          = EXCLUDED.width,
       height         = EXCLUDED.height`,
    [
      filename, currentHash,
      result.paperBoundary ? JSON.stringify(result.paperBoundary) : null,
      JSON.stringify(result.rings ?? []),
      JSON.stringify(arrowsFull),
      width, height,
    ],
  );

  const displayResult: ArcheryResult = {
    success: result.success, rings: result.rings ?? [],
    paperBoundary: result.paperBoundary, error: result.error,
  };
  return { filename, base64, width, height, result: displayResult, arrows: arrowsFull };
}

function splineCentroid(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

function splineRadius(ring: SplineRing): number {
  const [cx, cy] = splineCentroid(ring);
  const radii = ring.points.map(([x, y]) => Math.hypot(x - cx, y - cy));
  return radii.reduce((s, r) => s + r, 0) / radii.length;
}

function ringToPath(ring: SplineRing, N = 120): string {
  const pts = sampleClosedSpline(ring.points, N);
  const d = pts
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ') + ' Z';
  return d;
}

function renderSvg(
  base64: string,
  width: number,
  height: number,
  rings: SplineRing[],
  arrows: ScoredArrow[],
  paperBoundary?: TargetBoundary,
  ringPoints?: Pixel[][],
  rayDebug?: RayDebugEntry[],
): string {
  // Draw outermost rings first so inner rings render on top
  const pathsSvg = [...rings]
    .map((ring, idx) => ({ ring, idx }))
    .reverse()
    .map(({ ring, idx }) => {
      const color = RING_COLORS[idx] ?? '#FFFFFF';
      const d = ringToPath(ring);
      // White rings need a dark halo; black rings a light halo
      if (idx >= 8) {
        return (
          `<path d="${d}" fill="none" stroke="#000" stroke-width="4"/>` +
          `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`
        );
      }
      if (idx >= 6) {
        return (
          `<path d="${d}" fill="none" stroke="#aaa" stroke-width="4"/>` +
          `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`
        );
      }
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`;
    })
    .join('\n    ');

  // Target paper boundary — dashed lime polygon
  const boundarySvg = paperBoundary && paperBoundary.points.length >= 3
    ? (() => {
        const pts = paperBoundary.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        return `<polygon points="${pts}" fill="none" stroke="#00FF88" stroke-width="3" stroke-dasharray="12 6" opacity="0.85"/>`;
      })()
    : '';

  // Raw transition points — small dots in each ring's colour
  const pointsSvg = ringPoints
    ? ringPoints.map((pts, idx) => {
        if (pts.length === 0) return '';
        const color = RING_COLORS[idx] ?? '#00FF00';
        const stroke = (idx >= 6 && idx <= 7) ? '#fff' : '#000';
        return pts.map(p =>
          `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="2" fill="${color}" stroke="${stroke}" stroke-width="0.5" opacity="0.75"/>`
        ).join('');
      }).join('')
    : '';

  // Spline control points — diamond markers showing the sector-median points
  // that were actually used to build each SplineRing (K=12 per ring).
  // Only rendered for rings that have detected points (not interpolated rings [0,2,4,6,8]).
  const INTERPOLATED = new Set([0, 2, 4, 6, 8]);
  const ctrlSvg = rings.map((ring, idx) => {
    if (INTERPOLATED.has(idx)) return '';
    const color = RING_COLORS[idx] ?? '#00FF00';
    const S = 4; // half-size of diamond
    return ring.points.map(([x, y]) => {
      const d = `M${x},${y - S} L${x + S},${y} L${x},${y + S} L${x - S},${y} Z`;
      return `<path d="${d}" fill="${color}" stroke="#000" stroke-width="0.8" opacity="0.9"/>`;
    }).join('');
  }).join('');

  // Rays — line from centre to boundary with coloured tick marks at each detected transition.
  const rayDebugSvg = (() => {
    if (!rayDebug || rayDebug.length === 0 || rings.length === 0) return '';
    const [cx, cy] = splineCentroid(rings[1] ?? rings[0]);
    return rayDebug.map(({ theta, boundary, distances }) => {
      const cosT = Math.cos(theta), sinT = Math.sin(theta);
      const bx = (cx + boundary * cosT).toFixed(1);
      const by = (cy + boundary * sinT).toFixed(1);
      const deg = (theta * 180 / Math.PI).toFixed(1);
      const line = `<line x1="${cx.toFixed(1)}" y1="${cy.toFixed(1)}" x2="${bx}" y2="${by}" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>`;
      const ticks = distances.map((d, k) => {
        if (d === null || d <= 0) return '';
        const tx = (cx + d * cosT).toFixed(1);
        const ty = (cy + d * sinT).toFixed(1);
        const color = RING_COLORS[k] ?? '#fff';
        const stroke = (k >= 6 && k <= 7) ? '#fff' : '#000';
        return `<circle cx="${tx}" cy="${ty}" r="4" fill="${color}" stroke="${stroke}" stroke-width="1"/>`;
      }).join('');
      const label = `<text x="${bx}" y="${by}" font-size="10" fill="#fff" stroke="#000" stroke-width="2" paint-order="stroke" dx="4" dy="4">${deg}°</text>`;
      return line + ticks + label;
    }).join('');
  })();

  // Arrows — tip dot + score label
  const arrowsSvg = arrows.map((a) => {
    const [tx, ty] = a.tip;
    const scoreLabel = a.score === 'X' ? 'X' : String(a.score);
    const dot   = `<circle cx="${tx.toFixed(1)}" cy="${ty.toFixed(1)}" r="6" fill="#FF6600" stroke="#fff" stroke-width="1.5"/>`;
    const label = `<text x="${tx.toFixed(1)}" y="${(ty - 9).toFixed(1)}" font-size="13" font-weight="bold" fill="#FF6600" stroke="#000" stroke-width="2.5" paint-order="stroke" text-anchor="middle">${scoreLabel}</text>`;
    return dot + label;
  }).join('\n    ');

  return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="display:block;max-width:100%;height:auto">
    <image href="${base64}" width="${width}" height="${height}"/>
    ${boundarySvg}
    ${rayDebugSvg}
    ${pointsSvg}
    ${ctrlSvg}
    ${pathsSvg}
    ${arrowsSvg}
  </svg>`;
}

function renderRingTable(rings: SplineRing[], paperBoundary?: TargetBoundary): string {
  const rows = rings
    .map((ring, i) => {
      const score = 10 - i;
      const [cx, cy] = splineCentroid(ring);
      const r = splineRadius(ring);
      const prevR = i > 0 ? splineRadius(rings[i - 1]) : null;
      const ratio = prevR !== null ? (r / prevR).toFixed(3) : '—';
      return `<tr>
        <td>${i}</td><td>${score}</td>
        <td>${cx.toFixed(1)}</td><td>${cy.toFixed(1)}</td>
        <td>${r.toFixed(1)}</td><td>${ring.points.length}</td><td>${ratio}</td>
      </tr>`;
    })
    .join('\n');

  const boundaryRow = paperBoundary
    ? paperBoundary.points.map(([x, y], i) => {
        return `<tr style="${i === 0 ? 'border-top:2px solid #00FF88;' : ''}color:#00FF88">
          <td colspan="2">boundary v${i}</td>
          <td>${x.toFixed(0)}</td><td>${y.toFixed(0)}</td>
          <td colspan="3">—</td>
        </tr>`;
      }).join('\n')
    : '';

  return `<table>
    <thead><tr>
      <th>Ring</th><th>Score</th><th>cx</th><th>cy</th>
      <th>radius</th><th>pts</th><th>r-ratio</th>
    </tr></thead>
    <tbody>${rows}${boundaryRow}</tbody>
  </table>`;
}

function renderArrowTable(arrows: ScoredArrow[]): string {
  if (arrows.length === 0) return '<p style="color:#666;font-size:0.8rem;padding:4px 0">No arrows detected.</p>';
  const rows = arrows.map((a, i) => {
    const [tx, ty] = a.tip;
    const scoreLabel = a.score === 'X' ? 'X' : String(a.score);
    return `<tr>
      <td>${i + 1}</td>
      <td>${tx.toFixed(0)}, ${ty.toFixed(0)}</td>
      <td style="font-weight:bold;color:#FF6600">${scoreLabel}</td>
    </tr>`;
  }).join('\n');
  return `<table>
    <thead><tr><th>#</th><th>tip (x, y)</th><th>score</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// Canonical expected hue ranges per zone, for highlighting outliers
const ZONE_COLORS: Record<string, string> = {
  gold:  '#FFD700',
  red:   '#E8000D',
  blue:  '#006CB7',
  black: '#888888',
  white: '#FFFFFF',
};

function renderCalibrationTable(cal: ColourCalibration): string {
  const zones: (keyof ColourCalibration)[] = ['gold', 'red', 'blue', 'black', 'white'];
  const rows = zones.map(z => {
    const [h, s, v] = cal[z];
    const color = ZONE_COLORS[z];
    const swatch = `<span style="display:inline-block;width:16px;height:16px;border-radius:3px;background:${color};border:1px solid #555;vertical-align:middle"></span>`;
    return `<tr>
      <td>${swatch} ${z}</td>
      <td>${h.toFixed(1)}°</td>
      <td>${s.toFixed(3)}</td>
      <td>${v.toFixed(3)}</td>
    </tr>`;
  }).join('\n');

  return `<table>
    <thead><tr><th>Zone</th><th>H (corrected)</th><th>S</th><th>V</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

const HTML_HEAD = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ArcheryCounter — Detection Report</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #1a1a1a; color: #e0e0e0; font-family: system-ui, sans-serif; padding: 24px; }
    h1 { font-size: 1.6rem; margin-bottom: 6px; }
    .meta { color: #888; font-size: 0.85rem; margin-bottom: 32px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 24px; }
    .entry { background: #252525; border-radius: 10px; padding: 16px; border: 2px solid #383838; }
    .entry.success { border-color: #2a6b40; }
    .entry.error   { border-color: #7a2222; }
    h2 { font-size: 0.9rem; color: #aaa; margin-bottom: 6px; font-family: monospace; word-break: break-all; }
    .status { font-size: 0.85rem; margin-bottom: 12px; }
    .entry.success .status { color: #5cb878; }
    .entry.error   .status { color: #e05555; }
    .svg-wrapper { overflow: hidden; border-radius: 6px; line-height: 0; }
    details { margin-top: 14px; }
    summary { cursor: pointer; font-size: 0.82rem; color: #777; user-select: none; padding: 2px 0; }
    summary:hover { color: #ccc; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.78rem; }
    th { background: #2e2e2e; padding: 5px 8px; text-align: left; color: #999; font-weight: 600; }
    td { padding: 4px 8px; border-top: 1px solid #2e2e2e; font-family: monospace; color: #ccc; }
    tr:nth-child(even) td { background: #1e1e1e; }
  </style>
</head>
<body>
  <h1>ArcheryCounter &#8212; Detection Report</h1>`;

function renderSection(entry: ImageEntry): string {
  const { filename, base64, width, height, result, arrows } = entry;
  const statusClass = result.success ? 'success' : 'error';
  const statusText = result.success
    ? `&#10003; ${result.rings.length} rings &nbsp;&middot;&nbsp; ${arrows.length} arrow${arrows.length !== 1 ? 's' : ''} detected &nbsp;(${width}&times;${height} px)`
    : `&#10007; Detection failed: ${result.error ?? 'unknown error'}`;

  const content = result.success
    ? renderSvg(base64, width, height, result.rings, arrows, result.paperBoundary, result.ringPoints, result.rayDebug)
    : base64
      ? `<img src="${base64}" style="max-width:100%;border-radius:6px" alt="${filename}"/>`
      : `<p style="color:#888;padding:12px">Image could not be loaded.</p>`;

  const calibTable = result.success && result.calibration
    ? `<details><summary>Colour calibration</summary>${renderCalibrationTable(result.calibration)}</details>`
    : '';
  const ringTable = result.success
    ? `<details><summary>Ring data</summary>${renderRingTable(result.rings, result.paperBoundary)}</details>`
    : '';
  const arrowTable = result.success
    ? `<details open><summary>Arrows (${arrows.length})</summary>${renderArrowTable(arrows)}</details>`
    : '';

  return `
  <section class="entry ${statusClass}">
    <h2>${filename}</h2>
    <p class="status">${statusText}</p>
    <div class="svg-wrapper">${content}</div>
    ${arrowTable}
    ${calibTable}
    ${ringTable}
  </section>`;
}

async function main(): Promise<void> {
  const currentHash = computeAlgorithmHash();

  const jpgFiles = fs
    .readdirSync(IMAGES_DIR)
    .filter(f => /\.(jpg|jpeg)$/i.test(f))
    .map(f => path.join(IMAGES_DIR, f))
    .sort();

  if (jpgFiles.length === 0) {
    console.error(`No JPEG files found in ${IMAGES_DIR}`);
    process.exit(1);
  }

  console.log(`Algorithm hash: ${currentHash}`);
  console.log(`Processing ${jpgFiles.length} image(s) from ${IMAGES_DIR} (parallel)\n`);

  const entries: ImageEntry[] = [];
  for (const imgPath of jpgFiles) {
    const filename = path.basename(imgPath);
    try {
      const entry = await processImage(imgPath, currentHash);
      const label = entry.result.success ? 'ok' : `FAILED: ${entry.result.error}`;
      console.log(`  ${filename} ... ${label}`);
      entries.push(entry);
    } catch (err) {
      console.log(`  ${filename} ... EXCEPTION: ${err}`);
      entries.push({
        filename,
        base64: '',
        width: 0,
        height: 0,
        result: { success: false, rings: [], error: String(err) },
        arrows: [],
      } as ImageEntry);
    }
  }

  // Stream HTML to disk — avoids holding the full report string in memory.
  const passCount = entries.filter(e => e.result.success).length;
  const timestamp = new Date().toISOString();
  const out = fs.createWriteStream(OUTPUT_PATH, { encoding: 'utf8' });
  await new Promise<void>((resolve, reject) => {
    out.on('error', reject);
    out.write(HTML_HEAD);
    out.write(`\n  <p class="meta">Generated: ${timestamp} &nbsp;&middot;&nbsp; ${passCount}/${entries.length} images passed</p>`);
    out.write('\n  <div class="grid">');
    for (const entry of entries) out.write(renderSection(entry));
    out.write('\n  </div>\n</body>\n</html>\n');
    out.end(resolve);
  });

  console.log(`\nWrote: ${OUTPUT_PATH}`);
  console.log(`Result: ${passCount}/${entries.length} passed`);

  await db.end();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
