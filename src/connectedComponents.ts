/**
 * Flood-fill connected component labelling on a flat binary mask.
 *
 * @param mask   Flat array of length width*height. Non-zero = foreground.
 * @param width  Image width in pixels.
 * @param height Image height in pixels.
 * @returns Array of components, each as a sorted array of pixel indices.
 */
export function connectedComponents(
  mask: ArrayLike<number>,
  width: number,
  height: number,
): number[][] {
  const visited = new Uint8Array(width * height);
  const components: number[][] = [];

  for (let start = 0; start < width * height; start++) {
    if (!mask[start] || visited[start]) continue;

    // BFS flood-fill from this seed pixel
    const component: number[] = [];
    const queue: number[] = [start];
    visited[start] = 1;

    while (queue.length > 0) {
      const idx = queue.pop()!;
      component.push(idx);
      const x = idx % width;
      const y = (idx - x) / width;

      if (x > 0)          { const n = idx - 1;     if (mask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
      if (x < width - 1)  { const n = idx + 1;     if (mask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
      if (y > 0)          { const n = idx - width; if (mask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
      if (y < height - 1) { const n = idx + width; if (mask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
    }

    components.push(component);
  }

  return components;
}
