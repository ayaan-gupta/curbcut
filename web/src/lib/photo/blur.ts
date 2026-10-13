import type { Pixels } from './pixels'

export interface Region {
  x: number
  y: number
  width: number
  height: number
}

/**
 * Destroys the detail inside each region, in place.
 *
 * The region (grown by a margin, since detectors hug the face tightly) is
 * replaced by a coarse mosaic, at most three cells across, and the mosaic is
 * then smoothed. What is left is a handful of average colours. That is not a
 * Gaussian blur that could be sharpened back: the pixels are gone.
 */
export function blurRegions(photo: Pixels, regions: Region[], margin = 0.3): void {
  const { data, width, height } = photo
  for (const r of regions) {
    const x0 = Math.max(0, Math.floor(r.x - r.width * margin))
    const y0 = Math.max(0, Math.floor(r.y - r.height * margin))
    const x1 = Math.min(width, Math.ceil(r.x + r.width * (1 + margin)))
    const y1 = Math.min(height, Math.ceil(r.y + r.height * (1 + margin)))
    const w = x1 - x0
    const h = y1 - y0
    if (w <= 0 || h <= 0) continue
    const cell = Math.max(4, Math.ceil(Math.max(w, h) / 3))
    const cols = Math.ceil(w / cell)
    const rows = Math.ceil(h / cell)
    const means = new Float32Array(cols * rows * 3)
    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        let r0 = 0, g0 = 0, b0 = 0, n = 0
        for (let y = y0 + cy * cell; y < Math.min(y1, y0 + (cy + 1) * cell); y++) {
          for (let x = x0 + cx * cell; x < Math.min(x1, x0 + (cx + 1) * cell); x++) {
            const p = (y * width + x) * 4
            r0 += data[p]!; g0 += data[p + 1]!; b0 += data[p + 2]!; n++
          }
        }
        const m = (cy * cols + cx) * 3
        means[m] = r0 / n; means[m + 1] = g0 / n; means[m + 2] = b0 / n
      }
    }
    // Paint each pixel from the cell means, blended with the neighbouring cells so the mosaic has no hard edges to read.
    for (let y = y0; y < y1; y++) {
      const fy = Math.min(rows - 1, Math.max(0, (y - y0) / cell - 0.5))
      const cy0 = Math.floor(fy)
      const cy1 = Math.min(rows - 1, cy0 + 1)
      const ty = fy - cy0
      for (let x = x0; x < x1; x++) {
        const fx = Math.min(cols - 1, Math.max(0, (x - x0) / cell - 0.5))
        const cx0 = Math.floor(fx)
        const cx1 = Math.min(cols - 1, cx0 + 1)
        const tx = fx - cx0
        const p = (y * width + x) * 4
        for (let c = 0; c < 3; c++) {
          const top = means[(cy0 * cols + cx0) * 3 + c]! * (1 - tx) + means[(cy0 * cols + cx1) * 3 + c]! * tx
          const bottom = means[(cy1 * cols + cx0) * 3 + c]! * (1 - tx) + means[(cy1 * cols + cx1) * 3 + c]! * tx
          data[p + c] = Math.round(top * (1 - ty) + bottom * ty)
        }
      }
    }
  }
}
