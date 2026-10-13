/** Pixel work that has to give the same answer in the browser, in Node tests and in the Python trainer. */

export interface Pixels {
  /** RGBA, row-major, 4 bytes per pixel. */
  data: Uint8ClampedArray | Uint8Array
  width: number
  height: number
}

/**
 * Box-filter resize: each output pixel is the mean of the input pixels whose
 * centres fall inside its footprint. This is what Pillow's Image.BOX does,
 * which is what the trainer uses, so a photo is shrunk the same way on the
 * phone as it was during evaluation.
 */
export function resizeBox(src: Pixels, width: number, height: number): Pixels {
  const taps = (inSize: number, outSize: number) => {
    const scale = inSize / outSize
    const filter = Math.max(1, scale)
    const ranges: Array<[number, number]> = []
    for (let o = 0; o < outSize; o++) {
      const centre = (o + 0.5) * scale
      let lo = Math.max(0, Math.floor(centre - filter / 2))
      let hi = Math.min(inSize, Math.ceil(centre + filter / 2))
      while (lo < hi && (lo + 0.5 - centre) / filter <= -0.5) lo++
      while (hi > lo && (hi - 0.5 - centre) / filter > 0.5) hi--
      if (hi <= lo) { lo = Math.min(inSize - 1, Math.max(0, Math.floor(centre))); hi = lo + 1 }
      ranges.push([lo, hi])
    }
    return ranges
  }
  const xs = taps(src.width, width)
  const ys = taps(src.height, height)
  // Horizontal pass, rounded to bytes like Pillow, then vertical.
  const mid = new Uint8ClampedArray(width * src.height * 4)
  for (let y = 0; y < src.height; y++) {
    for (let x = 0; x < width; x++) {
      const [lo, hi] = xs[x]!
      let r = 0, g = 0, b = 0, a = 0
      for (let i = lo; i < hi; i++) {
        const p = (y * src.width + i) * 4
        r += src.data[p]!; g += src.data[p + 1]!; b += src.data[p + 2]!; a += src.data[p + 3]!
      }
      const n = hi - lo
      const q = (y * width + x) * 4
      mid[q] = Math.round(r / n); mid[q + 1] = Math.round(g / n); mid[q + 2] = Math.round(b / n); mid[q + 3] = Math.round(a / n)
    }
  }
  const out = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y++) {
    const [lo, hi] = ys[y]!
    const n = hi - lo
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0, a = 0
      for (let i = lo; i < hi; i++) {
        const p = (i * width + x) * 4
        r += mid[p]!; g += mid[p + 1]!; b += mid[p + 2]!; a += mid[p + 3]!
      }
      const q = (y * width + x) * 4
      out[q] = Math.round(r / n); out[q + 1] = Math.round(g / n); out[q + 2] = Math.round(b / n); out[q + 3] = Math.round(a / n)
    }
  }
  return { data: out, width, height }
}

export function crop(src: Pixels, left: number, top: number, width: number, height: number): Pixels {
  const out = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y++) {
    const from = ((top + y) * src.width + left) * 4
    out.set(src.data.subarray(from, from + width * 4), y * width * 4)
  }
  return { data: out, width, height }
}

/** RGB planes (channel, row, column) as float32, each value scaled by `scale` after subtracting `offset`. */
export function toPlanes(src: Pixels, offset = 0, scale = 1): Float32Array {
  const n = src.width * src.height
  const out = new Float32Array(3 * n)
  for (let i = 0; i < n; i++) {
    out[i] = (src.data[i * 4]! - offset) * scale
    out[n + i] = (src.data[i * 4 + 1]! - offset) * scale
    out[2 * n + i] = (src.data[i * 4 + 2]! - offset) * scale
  }
  return out
}
