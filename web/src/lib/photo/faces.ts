import type { Runner } from '../onnx'
import { crop, resizeBox, toPlanes, type Pixels } from './pixels'

/** A box in photo pixels. */
export interface Box {
  x: number
  y: number
  width: number
  height: number
  score: number
}

const NET_W = 320
const NET_H = 240
export const FACE_THRESHOLD = 0.6
const NMS_IOU = 0.3

function iou(a: Box, b: Box): number {
  const x0 = Math.max(a.x, b.x)
  const y0 = Math.max(a.y, b.y)
  const x1 = Math.min(a.x + a.width, b.x + b.width)
  const y1 = Math.min(a.y + a.height, b.y + b.height)
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
  return inter / (a.width * a.height + b.width * b.height - inter + 1e-9)
}

/** Greedy non-maximum suppression: keep the best box, drop the ones that overlap it, repeat. */
export function nms(boxes: Box[], threshold = NMS_IOU): Box[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score)
  const kept: Box[] = []
  for (const box of sorted) if (!kept.some((k) => iou(k, box) > threshold)) kept.push(box)
  return kept
}

/** Turns the detector's raw output for one region into boxes in photo pixels. */
export function decode(scores: ArrayLike<number>, boxes: ArrayLike<number>, region: { x: number; y: number; width: number; height: number }, threshold = FACE_THRESHOLD): Box[] {
  const out: Box[] = []
  for (let i = 0; i < scores.length / 2; i++) {
    const score = scores[i * 2 + 1]!
    if (score < threshold) continue
    const x0 = Math.min(1, Math.max(0, boxes[i * 4]!))
    const y0 = Math.min(1, Math.max(0, boxes[i * 4 + 1]!))
    const x1 = Math.min(1, Math.max(0, boxes[i * 4 + 2]!))
    const y1 = Math.min(1, Math.max(0, boxes[i * 4 + 3]!))
    if (x1 <= x0 || y1 <= y0) continue
    out.push({ x: region.x + x0 * region.width, y: region.y + y0 * region.height, width: (x1 - x0) * region.width, height: (y1 - y0) * region.height, score })
  }
  return out
}

/**
 * Finds faces so they can be blurred before a photo leaves the phone.
 *
 * The detector (UltraFace RFB-320, MIT) sees a 320 x 240 picture, in which a
 * passer-by across the street is only a few pixels tall. So on anything
 * larger than 640 px the photo is also cut into a 2 x 2 grid of overlapping
 * tiles and each tile is searched on its own. It returns boxes and nothing
 * else: there is no face recognition here, and no way to add it from this
 * model's output.
 */
export async function detectFaces(runner: Runner, photo: Pixels, threshold = FACE_THRESHOLD): Promise<Box[]> {
  const regions = [{ x: 0, y: 0, width: photo.width, height: photo.height }]
  if (photo.width > 640 || photo.height > 640) {
    const tw = Math.round(photo.width * 0.6)
    const th = Math.round(photo.height * 0.6)
    for (const x of [0, photo.width - tw]) for (const y of [0, photo.height - th]) regions.push({ x, y, width: tw, height: th })
  }
  const found: Box[] = []
  for (const region of regions) {
    const part = regions.length > 1 && region.width !== photo.width ? crop(photo, region.x, region.y, region.width, region.height) : photo
    const input = toPlanes(resizeBox(part, NET_W, NET_H), 127, 1 / 128)
    const out = await runner.run(input, [1, 3, NET_H, NET_W])
    found.push(...decode(out.scores!.data as ArrayLike<number>, out.boxes!.data as ArrayLike<number>, region, threshold))
  }
  return nms(found)
}
