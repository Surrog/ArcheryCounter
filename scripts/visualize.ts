import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, EllipseData, ArcheryResult } from '../src/targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const OUTPUT_PATH = path.resolve(__dirname, '../report.html');

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
}

async function processImage(imgPath: string): Promise<ImageEntry> {
  const { Jimp } = require('jimp');
  const filename = path.basename(imgPath);

  // Load and scale for the algorithm (matches loadImageNode behaviour)
  const { rgba, width, height } = await loadImageNode(imgPath);

  // Load with Jimp again to export the *scaled* JPEG as a base64 data URL for
  // the HTML <image> tag.  The viewBox uses the same scaled dimensions, so
  // the SVG ellipses line up correctly.
  const img = await Jimp.read(imgPath);
  img.scaleToFit({ w: 1200, h: 1200 });
  const base64 = await img.getBase64('image/jpeg');

  const result = findTarget(rgba, width, height);
  return { filename, base64, width, height, result };
}

function renderSvg(
  base64: string,
  width: number,
  height: number,
  rings: EllipseData[],
): string {
  // Draw outermost rings first so inner rings render on top
  const ellipsesSvg = [...rings]
    .map((ring, idx) => ({ ring, idx }))
    .reverse()
    .map(({ ring, idx }) => {
      const rx = (ring.width / 2).toFixed(1);
      const ry = (ring.height / 2).toFixed(1);
      const cx = ring.centerX.toFixed(1);
      const cy = ring.centerY.toFixed(1);
      const angle = ring.angle.toFixed(1);
      const color = RING_COLORS[idx] ?? '#FFFFFF';
      const transform = `rotate(${angle},${cx},${cy})`;

      // White rings need a dark halo to be visible on bright backgrounds
      if (idx >= 8) {
        return (
          `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="${transform}" fill="none" stroke="#000" stroke-width="4"/>` +
          `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="${transform}" fill="none" stroke="${color}" stroke-width="2"/>`
        );
      }
      // Black rings get a light halo
      if (idx >= 6) {
        return (
          `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="${transform}" fill="none" stroke="#aaa" stroke-width="4"/>` +
          `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="${transform}" fill="none" stroke="${color}" stroke-width="2"/>`
        );
      }
      return `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="${transform}" fill="none" stroke="${color}" stroke-width="2"/>`;
    })
    .join('\n    ');

  // Small marker at the bullseye centre
  const bull = rings[0];
  const bullMarker = bull
    ? `<circle cx="${bull.centerX.toFixed(1)}" cy="${bull.centerY.toFixed(1)}" r="5" fill="#FFD700" stroke="#000" stroke-width="1"/>`
    : '';

  return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="display:block;max-width:100%;height:auto">
    <image href="${base64}" width="${width}" height="${height}"/>
    ${ellipsesSvg}
    ${bullMarker}
  </svg>`;
}

function renderRingTable(rings: EllipseData[]): string {
  const rows = rings
    .map((r, i) => {
      const score = 10 - i;
      return `<tr>
        <td>${i}</td><td>${score}</td>
        <td>${r.centerX.toFixed(1)}</td><td>${r.centerY.toFixed(1)}</td>
        <td>${r.width.toFixed(1)}</td><td>${r.height.toFixed(1)}</td>
        <td>${r.angle.toFixed(1)}</td>
      </tr>`;
    })
    .join('\n');
  return `<table>
    <thead><tr>
      <th>Ring</th><th>Score</th><th>cx</th><th>cy</th>
      <th>width</th><th>height</th><th>angle°</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function generateHtml(entries: ImageEntry[]): string {
  const passCount = entries.filter(e => e.result.success).length;
  const timestamp = new Date().toISOString();

  const sections = entries
    .map(({ filename, base64, width, height, result }) => {
      const statusClass = result.success ? 'success' : 'error';
      const statusText = result.success
        ? `&#10003; 10 rings detected &nbsp;(${width}&times;${height} px)`
        : `&#10007; Detection failed: ${result.error ?? 'unknown error'}`;

      const content = result.success
        ? renderSvg(base64, width, height, result.rings)
        : base64
          ? `<img src="${base64}" style="max-width:100%;border-radius:6px" alt="${filename}"/>`
          : `<p style="color:#888;padding:12px">Image could not be loaded.</p>`;

      const table = result.success
        ? `<details><summary>Ring data</summary>${renderRingTable(result.rings)}</details>`
        : '';

      return `
  <section class="entry ${statusClass}">
    <h2>${filename}</h2>
    <p class="status">${statusText}</p>
    <div class="svg-wrapper">${content}</div>
    ${table}
  </section>`;
    })
    .join('\n');

  return `<!DOCTYPE html>
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
  <h1>ArcheryCounter &#8212; Detection Report</h1>
  <p class="meta">Generated: ${timestamp} &nbsp;&middot;&nbsp; ${passCount}/${entries.length} images passed</p>
  <div class="grid">
${sections}
  </div>
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
