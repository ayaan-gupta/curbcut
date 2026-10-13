import type { Runner } from '../onnx'
import { crop, resizeBox, toPlanes, type Pixels } from './pixels'

export const PHOTO_CLASSES = ['present', 'missing', 'damaged', 'blocked'] as const
export type PhotoClass = (typeof PHOTO_CLASSES)[number]
export const INPUT = 224

export interface PhotoVerdict {
  label: PhotoClass
  confidence: number
  probs: Record<PhotoClass, number>
}

/** Shorter side to 224 by area averaging, centre crop. Same as deploy_preprocess in ml/photo/dataset.py. */
export function prepare(photo: Pixels): Float32Array {
  const scale = INPUT / Math.min(photo.width, photo.height)
  const w = Math.max(INPUT, Math.floor(photo.width * scale + 0.5))
  const h = Math.max(INPUT, Math.floor(photo.height * scale + 0.5))
  const small = resizeBox(photo, w, h)
  return toPlanes(crop(small, (w - INPUT) >> 1, (h - INPUT) >> 1, INPUT, INPUT))
}

/** Runs the curb classifier. Normalisation and softmax live inside the model file. */
export async function classifyPhoto(runner: Runner, photo: Pixels): Promise<PhotoVerdict> {
  const out = await runner.run(prepare(photo), [1, 3, INPUT, INPUT])
  const values = Array.from((out.probs ?? Object.values(out)[0])!.data as ArrayLike<number>, Number)
  let best = 0
  values.forEach((v, i) => { if (v > values[best]!) best = i })
  const probs = Object.fromEntries(PHOTO_CLASSES.map((c, i) => [c, values[i]!])) as Record<PhotoClass, number>
  return { label: PHOTO_CLASSES[best]!, confidence: values[best]!, probs }
}
