/** Milestone 3: on-device photo handling. Face finding, blurring, and the pixel pipeline the classifier depends on. */
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { blurRegions } from '../src/lib/photo/blur'
import { decode, detectFaces, nms, type Box } from '../src/lib/photo/faces'
import { crop, resizeBox, toPlanes, type Pixels } from '../src/lib/photo/pixels'
import { readFileSync } from 'node:fs'
import { classifyPhoto, PHOTO_CLASSES } from '../src/lib/photo/classify'
import { loadPixels, onCanvas } from './helpers/image'
import { nodeRunner, REPO } from './helpers/nodeRunner'

const CREW = path.join(__dirname, 'fixtures', 'apollo11-crew.jpg')

function variance(photo: Pixels, box: { x: number; y: number; width: number; height: number }): number {
  let sum = 0, sq = 0, n = 0
  for (let y = Math.floor(box.y); y < box.y + box.height; y++) {
    for (let x = Math.floor(box.x); x < box.x + box.width; x++) {
      const p = (y * photo.width + x) * 4
      const grey = (photo.data[p]! + photo.data[p + 1]! + photo.data[p + 2]!) / 3
      sum += grey; sq += grey * grey; n++
    }
  }
  return sq / n - (sum / n) ** 2
}

describe('pixels', () => {
  it('box-resizes by averaging exactly the pixels under each output pixel', () => {
    const src: Pixels = { width: 4, height: 2, data: Uint8ClampedArray.from([0, 0, 0, 255, 100, 100, 100, 255, 200, 0, 0, 255, 0, 200, 0, 255, 0, 0, 0, 255, 100, 100, 100, 255, 200, 0, 0, 255, 0, 200, 0, 255]) }
    const out = resizeBox(src, 2, 1)
    expect(Array.from(out.data)).toEqual([50, 50, 50, 255, 100, 100, 0, 255])
    // Upscaling repeats pixels instead of inventing values.
    const up = resizeBox({ width: 1, height: 1, data: Uint8ClampedArray.from([9, 8, 7, 255]) }, 3, 2)
    expect(Array.from(up.data.slice(0, 4))).toEqual([9, 8, 7, 255])
    expect(up.data.length).toBe(24)
  })

  it('crops and lays pixels out as channel planes', () => {
    const src: Pixels = { width: 2, height: 2, data: Uint8ClampedArray.from([1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]) }
    expect(Array.from(crop(src, 1, 0, 1, 2).data)).toEqual([4, 5, 6, 255, 10, 11, 12, 255])
    expect(Array.from(toPlanes(src))).toEqual([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12])
    expect(Array.from(toPlanes(src, 1, 0.5)).slice(0, 2)).toEqual([0, 1.5])
  })
})

describe('face boxes', () => {
  it('decodes scores and corner boxes into photo pixels and drops weak ones', () => {
    const scores = [0.1, 0.9, 0.8, 0.2, 0.3, 0.7]
    const boxes = [0.1, 0.2, 0.3, 0.6, 0, 0, 1, 1, 0.5, 0.5, 0.4, 0.9]
    const out = decode(scores, boxes, { x: 100, y: 50, width: 200, height: 100 }, 0.6)
    expect(out).toEqual([{ x: 120, y: 70, width: 40, height: 40, score: 0.9 }])
  })

  it('keeps the strongest of overlapping boxes', () => {
    const a: Box = { x: 0, y: 0, width: 10, height: 10, score: 0.8 }
    const b: Box = { x: 1, y: 1, width: 10, height: 10, score: 0.95 }
    const c: Box = { x: 40, y: 40, width: 10, height: 10, score: 0.7 }
    expect(nms([a, b, c])).toEqual([b, c])
  })
})

describe('the face detector on a real photograph', () => {
  it('finds the three faces in a public domain crew portrait', async () => {
    const runner = await nodeRunner('face-ultraface-320.onnx')
    const photo = await loadPixels(CREW)
    const faces = await detectFaces(runner, photo)
    expect(faces.length).toBe(3)
    for (const f of faces) {
      expect(f.score).toBeGreaterThan(0.9)
      expect(f.width).toBeGreaterThan(40)
      expect(f.y + f.height).toBeLessThan(photo.height * 0.6)
    }
  })

  it('finds them when they are small in a large frame, which needs the tiled pass', async () => {
    const runner = await nodeRunner('face-ultraface-320.onnx')
    const photo = await onCanvas(CREW, { width: 1280, height: 960 }, { left: 820, top: 520, width: 300 })
    const faces = await detectFaces(runner, photo)
    expect(faces.length).toBe(3)
    for (const f of faces) {
      expect(f.x).toBeGreaterThan(820)
      expect(f.y).toBeGreaterThan(520)
      expect(f.width).toBeLessThan(60)
    }
  })

  it('blurring a found face leaves nothing for the detector to find and flattens the pixels', async () => {
    const runner = await nodeRunner('face-ultraface-320.onnx')
    const photo = await loadPixels(CREW)
    const faces = await detectFaces(runner, photo)
    const before = faces.map((f) => variance(photo, f))
    const untouched = photo.data.slice(photo.data.length - 4000)
    blurRegions(photo, faces)
    faces.forEach((f, i) => expect(variance(photo, f)).toBeLessThan(before[i]! * 0.5))
    expect(await detectFaces(runner, photo)).toEqual([])
    // Pixels outside the regions are not touched.
    expect(Array.from(photo.data.slice(photo.data.length - 4000))).toEqual(Array.from(untouched))
  })

  it('blurs a hand-drawn region the same way, for licence plates', () => {
    const photo: Pixels = { width: 64, height: 64, data: new Uint8ClampedArray(64 * 64 * 4) }
    for (let i = 0; i < 64 * 64; i++) photo.data.set([(i * 37) % 256, (i * 91) % 256, (i * 13) % 256, 255], i * 4)
    const region = { x: 16, y: 16, width: 24, height: 12 }
    const before = variance(photo, region)
    blurRegions(photo, [region], 0)
    expect(variance(photo, region)).toBeLessThan(before * 0.05)
    expect(photo.data[3]).toBe(255)
  })
})

describe('the curb classifier on a real photograph', () => {
  it('gives the same probabilities in TypeScript as the Python pipeline that measured its accuracy', async () => {
    const want = JSON.parse(readFileSync(path.join(REPO, 'ml', 'photo', 'fixtures', 'sample_probs.json'), 'utf8')) as { classes: string[]; probs: number[] }
    expect(want.classes).toEqual([...PHOTO_CLASSES])
    const photo = await loadPixels(path.join(REPO, 'web', 'public', 'samples', 'curb-cut.jpg'))
    const verdict = await classifyPhoto(await nodeRunner('curb-photo.onnx'), photo)
    // Different JPEG decoders and resize rounding move a pixel by a level or two, so allow a few points.
    PHOTO_CLASSES.forEach((c, i) => expect(Math.abs(verdict.probs[c] - want.probs[i]!)).toBeLessThan(0.05))
    expect(verdict.label).toBe('present')
    expect(Object.values(verdict.probs).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 3)
  })
})
