/** Milestone 2: the roughness classifier as it runs on the phone. */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { predict, WalkClassifier } from '../src/lib/roughness/classifier'
import { ALL_FEATURE_NAMES, assemble, extractBase, relative, WINDOW, type MotionWindow } from '../src/lib/roughness/features'
import { Recorder, positionAt, selectTaggedPoints, type WindowResult } from '../src/lib/sensors/recorder'
import { Resampler } from '../src/lib/sensors/resample'
import { nodeRunner, REPO } from './helpers/nodeRunner'

interface Fixture {
  feature_names: string[]
  windows: number[][][]
  features: number[][]
  probabilities: number[][]
  labels: number[]
}
const fixture = JSON.parse(readFileSync(path.join(REPO, 'ml', 'roughness', 'fixtures', 'feature_parity.json'), 'utf8')) as Fixture

function toWindow(rows: number[][]): MotionWindow {
  const col = (i: number) => rows.map((r) => r[i]!)
  return { ax: col(0), ay: col(1), az: col(2), gx: col(3), gy: col(4), gz: col(5) }
}

describe('feature extraction', () => {
  it('lists the same features in the same order as the Python side', () => {
    expect(ALL_FEATURE_NAMES).toEqual(fixture.feature_names)
  })

  it('reproduces the Python features for a simulated walk to within float32 rounding', () => {
    const bases = fixture.windows.map((w) => extractBase(toWindow(w)))
    bases.forEach((base, i) => {
      const full = assemble(base, bases.slice(0, i), bases[i - 1] ?? null, bases[i + 1] ?? null)
      const want = fixture.features[i]!
      expect(full.length).toBe(want.length)
      full.forEach((value, j) => {
        const tolerance = 2e-3 * Math.max(1, Math.abs(want[j]!))
        expect(Math.abs(value - want[j]!), `window ${i}, ${ALL_FEATURE_NAMES[j]}`).toBeLessThan(tolerance)
      })
    })
  })

  it('does not care which way up the phone is', () => {
    const rows = fixture.windows[2]!
    // Rotate the device frame: x to y, y to z, z to x, and flip the sign of everything (iOS against Android).
    const turned = rows.map(([ax, ay, az, gx, gy, gz]) => [-az!, -ax!, -ay!, -gz!, -gx!, -gy!])
    const a = extractBase(toWindow(rows))
    const b = extractBase(toWindow(turned))
    a.forEach((value, j) => expect(Math.abs(value - b[j]!)).toBeLessThan(1e-3 * Math.max(1, Math.abs(value))))
  })

  it('keeps the relative features neutral until the walk has a baseline', () => {
    const bases = fixture.windows.map((w) => extractBase(toWindow(w)))
    expect(relative(bases[0]!, [])).toEqual([1, 1, 0])
    expect(relative(bases[2]!, bases.slice(0, 2))).toEqual([1, 1, 0])
    expect(relative(bases[4]!, bases.slice(0, 4))[0]).not.toBe(1)
  })
})

describe('the exported ONNX model', () => {
  it('gives the probabilities recorded when it was trained', async () => {
    const runner = await nodeRunner('roughness.onnx')
    for (let i = 0; i < fixture.features.length; i++) {
      const { probs } = await predict(runner, Float32Array.from(fixture.features[i]!))
      probs.forEach((p, j) => expect(Math.abs(p - fixture.probabilities[i]![j]!)).toBeLessThan(1e-4))
      expect(probs.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 4)
    }
  })

  it('classifies a walk one window late and matches the batch result', async () => {
    const runner = await nodeRunner('roughness.onnx')
    const walk = new WalkClassifier<number>(runner)
    const results = []
    for (let i = 0; i < fixture.windows.length; i++) {
      const done = await walk.push(toWindow(fixture.windows[i]!), i)
      if (i === 0) expect(done).toBeNull()
      else results.push(done!)
    }
    results.push((await walk.flush())!)
    expect(results.map((r) => r.meta)).toEqual(fixture.windows.map((_, i) => i))
    results.forEach((r, i) => r.probs.forEach((p, j) => expect(Math.abs(p - fixture.probabilities[i]![j]!)).toBeLessThan(2e-3)))
    expect(await walk.flush()).toBeNull()
  })
})

describe('resampling and recording', () => {
  it('turns a jittery 60 Hz stream into exact 50 Hz windows', () => {
    const resampler = new Resampler()
    const windows = []
    let t = 1000
    for (let i = 0; i < 400; i++) {
      t += 1000 / 60 + (i % 3 === 0 ? 1.5 : -0.75)
      const out = resampler.push({ t, ax: Math.sin(t / 100), ay: 0, az: 9.81, gx: 0, gy: 0, gz: 0 })
      windows.push(...out.windows)
      expect(out.gap).toBe(false)
    }
    expect(windows.length).toBe(3)
    for (const w of windows) {
      expect(w.ax.length).toBe(WINDOW)
      expect(w.tEnd - w.tStart).toBeCloseTo(2000, 6)
    }
    // Interpolated values follow the underlying signal.
    const w = windows[1]!
    expect(Math.abs((w.ax[50] as number) - Math.sin((w.tStart + 50 * 20) / 100))).toBeLessThan(0.02)
    expect(windows[1]!.tStart).toBeCloseTo(windows[0]!.tEnd, 6)
  })

  it('drops the partial window when the sensor stream stalls', () => {
    const resampler = new Resampler()
    let windows = 0
    for (let i = 0; i < 80; i++) windows += resampler.push({ t: i * 20, ax: 0, ay: 0, az: 9.8, gx: 0, gy: 0, gz: 0 }).windows.length
    const stalled = resampler.push({ t: 80 * 20 + 900, ax: 0, ay: 0, az: 9.8, gx: 0, gy: 0, gz: 0 })
    expect(stalled.gap).toBe(true)
    for (let i = 1; i <= 99; i++) windows += resampler.push({ t: 80 * 20 + 900 + i * 20, ax: 0, ay: 0, az: 9.8, gx: 0, gy: 0, gz: 0 }).windows.length
    expect(windows).toBe(1)
  })

  it('interpolates position between GPS fixes and gives up when they are too far apart in time', () => {
    const trace = [{ t: 0, lon: 0, lat: 0, accuracy: 4 }, { t: 2000, lon: 0.002, lat: 0.001, accuracy: 6 }, { t: 30_000, lon: 1, lat: 1, accuracy: 5 }]
    const mid = positionAt(trace, 500)!
    expect(mid.lon).toBeCloseTo(0.0005, 9)
    expect(mid.accuracy).toBe(6)
    expect(positionAt(trace, 15_000)).toBeNull()
    expect(positionAt([], 5)).toBeNull()
    expect(positionAt(trace, 33_000)!.lon).toBe(1)
  })

  it('records a simulated walk end to end without touching the network', async () => {
    const runner = await nodeRunner('roughness.onnx')
    const seen: WindowResult[] = []
    const recorder = new Recorder(runner, (w) => seen.push(w))
    let t = 0
    fixture.windows.forEach((rows, k) => {
      recorder.fix({ t, lon: -122.32 + k * 0.00003, lat: 47.616, accuracy: 5 })
      for (const [ax, ay, az, gx, gy, gz] of rows) {
        recorder.motion({ t, ax: ax!, ay: ay!, az: az!, gx: gx!, gy: gy!, gz: gz! })
        t += 20
      }
    })
    recorder.fix({ t, lon: -122.32 + 8 * 0.00003, lat: 47.616, accuracy: 5 })
    const session = await recorder.stop()
    expect(session.windows.length).toBe(8)
    expect(seen.length).toBe(8)
    expect(session.windows.every((w) => w.position !== null)).toBe(true)
    expect(session.windows.map((w) => w.label)).toContain('bump')
  })
})

describe('choosing what to upload', () => {
  const at = (metresEast: number, label: WindowResult['label'], confidence = 0.9, accuracy = 5): WindowResult => ({
    tStart: metresEast * 1000,
    tEnd: metresEast * 1000 + 2000,
    label,
    probs: [0, 0, 0],
    confidence,
    peak: 12.34,
    position: { t: 0, lon: -122.32 + metresEast / 75_000, lat: 47.616, accuracy },
    raw: {} as WindowResult['raw'],
  })

  it('uploads nothing for a walk over smooth ground', () => {
    expect(selectTaggedPoints([at(0, 'smooth'), at(3, 'smooth'), at(6, 'smooth')])).toEqual([])
  })

  it('reports rough ground once every 6 m and every bump', () => {
    const points = selectTaggedPoints([at(0, 'rough'), at(2, 'rough'), at(4, 'rough'), at(7, 'rough'), at(8, 'bump'), at(9, 'bump')])
    expect(points.map((p) => p.state)).toEqual(['rough', 'rough', 'bump', 'bump'])
    expect(points[2]!.value).toBe(12.3)
  })

  it('skips windows the model is unsure of or GPS could not place', () => {
    const noFix = { ...at(0, 'bump'), position: null }
    expect(selectTaggedPoints([at(0, 'bump', 0.5), at(10, 'rough', 0.9, 40), noFix])).toEqual([])
  })

  it('sends a smooth pass only as counter-evidence next to a known surface issue', () => {
    const known: [number, number][] = [[-122.32 + 20 / 75_000, 47.616]]
    const points = selectTaggedPoints([at(0, 'smooth'), at(19, 'smooth'), at(21, 'smooth'), at(60, 'smooth')], known)
    expect(points.length).toBe(1)
    expect(points[0]!.state).toBe('smooth')
  })
})
