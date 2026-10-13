import type { Pixels } from './pixels'

const MAX_SIDE = 1280

/** Decodes an image file into pixels, no larger than 1280 px on its long side. Drawing through a canvas also drops EXIF. */
export async function fileToPixels(file: Blob): Promise<Pixels> {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height))
  const width = Math.round(bitmap.width * scale)
  const height = Math.round(bitmap.height * scale)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bitmap, 0, 0, width, height)
  bitmap.close()
  const image = ctx.getImageData(0, 0, width, height)
  return { data: image.data, width, height }
}

export function drawPixels(canvas: HTMLCanvasElement, photo: Pixels): void {
  canvas.width = photo.width
  canvas.height = photo.height
  canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(photo.data), photo.width, photo.height), 0, 0)
}

export function pixelsToJpeg(photo: Pixels, quality = 0.82): Promise<Blob> {
  const canvas = document.createElement('canvas')
  drawPixels(canvas, photo)
  return new Promise((resolve, reject) => canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('could not encode the photo'))), 'image/jpeg', quality))
}
