import gifenc from 'gifenc'

// gifenc is CommonJS, and Node's ESM loader cannot see its named exports.
const { applyPalette, GIFEncoder, quantize } = gifenc

export interface Frame {
  rgba: Uint8Array
  /** Milliseconds this frame stays on screen. */
  delay: number
}

/**
 * Encodes frames as a GIF with one shared palette and inter-frame
 * differencing: a pixel that did not change since the previous frame is
 * written as transparent. Screen recordings are mostly still, so this cuts
 * the file to a fraction of what whole frames would cost.
 */
export function encodeGif(frames: Frame[], width: number, height: number): Uint8Array {
  // Build the palette from an evenly spaced sample of pixels across all frames.
  const step = Math.max(1, Math.floor((frames.length * width * height) / 400_000))
  const sample: number[] = []
  let n = 0
  for (const f of frames)
    for (let i = 0; i < f.rgba.length; i += 4) {
      if (n++ % step === 0) sample.push(f.rgba[i]!, f.rgba[i + 1]!, f.rgba[i + 2]!, 255)
    }
  const palette = quantize(new Uint8Array(sample), 255)
  const transparentIndex = palette.length
  const withGap = [...palette, [0, 0, 0]]

  const gif = GIFEncoder()
  let previous: Uint8Array | null = null
  frames.forEach((frame, i) => {
    const index = applyPalette(frame.rgba, palette)
    const out = new Uint8Array(index)
    if (previous) for (let p = 0; p < out.length; p++) if (index[p] === previous[p]) out[p] = transparentIndex
    gif.writeFrame(out, width, height, {
      palette: i === 0 ? withGap : undefined,
      delay: frame.delay,
      transparent: i > 0,
      transparentIndex,
      dispose: 1,
      repeat: 0,
    })
    previous = index
  })
  gif.finish()
  return gif.bytes()
}
