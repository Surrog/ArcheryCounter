/**
 * Moore-neighbor contour tracing for a binary mask.
 *
 * Given a flat binary mask and a list of pixel indices belonging to one
 * connected component, returns the outer contour as an ordered array of
 * [x, y] pixel coordinates.
 *
 * Uses the Jacob's stopping criterion: stop when the start pixel is visited
 * a second time entering from the same direction as the first entry.
 */

// 8-connected Moore neighbourhood offsets, clockwise from right
const DX = [ 1,  1,  0, -1, -1, -1,  0,  1];
const DY = [ 0,  1,  1,  1,  0, -1, -1, -1];

export function traceContour(
  mask: ArrayLike<number>,
  width: number,
  height: number,
  componentPixels: number[],
): [number, number][] {
  if (componentPixels.length === 0) return [];

  // Find the topmost-leftmost pixel of the component as the start pixel
  let startIdx = componentPixels[0];
  for (const idx of componentPixels) {
    const sy = Math.floor(startIdx / width);
    const iy = Math.floor(idx / width);
    if (iy < sy || (iy === sy && idx % width < startIdx % width)) {
      startIdx = idx;
    }
  }

  const startX = startIdx % width;
  const startY = (startIdx - startX) / width;

  // Single-pixel component
  if (componentPixels.length === 1) return [[startX, startY]];

  const contour: [number, number][] = [];
  let cx = startX, cy = startY;
  // Entry direction into the start pixel — from the left (direction index 4 = left)
  let entryDir = 4;
  const startEntryDir = entryDir;

  let iterations = 0;
  const maxIter = componentPixels.length * 8 + 8;

  do {
    contour.push([cx, cy]);

    // Search Moore neighbourhood clockwise, starting from the pixel we came from
    const backDir = (entryDir + 4) % 8;  // direction back to previous pixel
    let found = false;
    for (let d = 1; d <= 8; d++) {
      const dir = (backDir + d) % 8;
      const nx = cx + DX[dir];
      const ny = cy + DY[dir];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      if (mask[ny * width + nx]) {
        entryDir = (dir + 4) % 8;  // how we entered nx,ny
        cx = nx;
        cy = ny;
        found = true;
        break;
      }
    }

    if (!found) break; // isolated pixel surrounded by background

    iterations++;
    if (iterations > maxIter) break; // safety
  } while (!(cx === startX && cy === startY && entryDir === startEntryDir));

  return contour;
}
