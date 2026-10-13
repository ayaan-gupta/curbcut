import sharp from 'sharp'
import type { Pixels } from '../../src/lib/photo/pixels'

export async function loadPixels(file: string): Promise<Pixels> {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), width: info.width, height: info.height }
}

/** Pastes `src` into a larger grey canvas at a smaller size: a stand-in for faces far from the camera. */
export async function onCanvas(file: string, canvas: { width: number; height: number }, at: { left: number; top: number; width: number }): Promise<Pixels> {
  const small = await sharp(file).resize({ width: at.width }).toBuffer()
  const { data, info } = await sharp({ create: { width: canvas.width, height: canvas.height, channels: 3, background: { r: 120, g: 124, b: 128 } } })
    .composite([{ input: small, left: at.left, top: at.top }])
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
  return { data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), width: info.width, height: info.height }
}
