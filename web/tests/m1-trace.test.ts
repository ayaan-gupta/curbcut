/** Milestone 1: the GPS trace and the tagged points that go on the map. */
import { describe, expect, it } from 'vitest'
import { exportWindows, labelWindow } from '../src/lib/collect'
import { formatDistance, haversine, pathLength, travelled, type LonLat } from '../src/lib/geo'
import { Recorder, selectTaggedPoints } from '../src/lib/sensors/recorder'
import type { TimedWindow } from '../src/lib/sensors/resample'

const east = (metres: number): LonLat => [-122.32 + metres / 75_040, 47.616]

describe('trace geometry', () => {
  it('measures distance in metres', () => {
    expect(haversine(east(0), east(100))).toBeCloseTo(100, 0)
    expect(pathLength([east(0), east(30), east(10)])).toBeCloseTo(50, 0)
    expect(formatDistance(149.4)).toBe('149 m')
    expect(formatDistance(1260)).toBe('1.3 km')
  })

  it('does not let GPS jitter inflate the distance travelled', () => {
    // 150 m in a straight line, one fix per 1.25 m, each off by up to 2 m sideways and lengthways.
    const noisy: LonLat[] = []
    for (let i = 0; i <= 120; i++) {
      const wobble = ((i * 7919) % 400) / 100 - 2
      noisy.push([east(i * 1.25 + wobble)[0], 47.616 + (((i * 104729) % 400) / 100 - 2) / 111_320])
    }
    expect(pathLength(noisy)).toBeGreaterThan(220)
    const d = travelled(noisy)
    expect(d).toBeGreaterThan(135)
    expect(d).toBeLessThan(170)
  })
})

describe('a recording session', () => {
  it('keeps the trace on the device and tags only the windows worth reporting', async () => {
    // A stand-in model, so this test is about the trace and the tagging, not about the classifier:
    // it calls a window a bump when its peak vertical acceleration (feature 2) is over 5 m/s^2.
    const model = { run: async (f: Float32Array) => ({ probabilities: { data: f[2]! > 5 ? [0.05, 0.05, 0.9] : [0.9, 0.05, 0.05], dims: [1, 3] } }) }
    const recorder = new Recorder(model)
    // Four seconds of a phone riding smoothly, then four with a hard jolt every 2 s, moving east at 1 m/s.
    for (let i = 0; i < 400; i++) {
      const t = i * 20
      if (i % 50 === 0) recorder.fix({ t, lon: east(i / 50)[0], lat: 47.616, accuracy: 4 })
      // A jolt that rings for a third of a second, the way a wheel dropping off a lip does.
      const since = ((i % 100) - 40) / 50
      const jolt = i >= 200 && since >= 0 && since < 0.4 ? 13 * Math.exp(-since / 0.07) * Math.cos(2 * Math.PI * 9 * since) : 0
      recorder.motion({ t, ax: 0.02 * Math.sin(i), ay: 0, az: 9.81 + jolt + 0.03 * Math.cos(i * 1.7), gx: 0, gy: 0, gz: 0 })
    }
    const session = await recorder.stop()
    expect(session.trace.length).toBe(8)
    expect(session.windows.length).toBe(4)
    expect(session.windows.slice(0, 2).map((w) => w.label)).toEqual(['smooth', 'smooth'])
    expect(session.windows.slice(2).map((w) => w.label)).toEqual(['bump', 'bump'])
    const tagged = selectTaggedPoints(session.windows)
    expect(tagged.map((p) => p.state)).toEqual(['bump', 'bump'])
    // A tagged point sits on the trace, at the middle of its window.
    expect(haversine([tagged[0]!.lon, tagged[0]!.lat], east(5))).toBeLessThan(0.5)
    expect(tagged[0]!.value).toBeGreaterThan(8)
  })

  it('drops GPS fixes too vague to place anything', () => {
    const recorder = new Recorder({ run: async () => ({}) })
    recorder.fix({ t: 0, lon: 0, lat: 0, accuracy: 5 })
    recorder.fix({ t: 1000, lon: 0, lat: 0, accuracy: 300 })
    expect(recorder.trace.length).toBe(1)
  })
})

describe('labelled windows for retraining', () => {
  it('exports the format ml/roughness/retrain.py reads', () => {
    const raw = { ax: Array(100).fill(0.123456), ay: Array(100).fill(0), az: Array(100).fill(9.81), gx: Array(100).fill(0), gy: Array(100).fill(0), gz: Array(100).fill(0), tStart: 0, tEnd: 2000 } as TimedWindow
    const doc = JSON.parse(exportWindows([labelWindow(raw, 'rough', 'abc123', 4)])) as { format: string; rate_hz: number; window: number; windows: Array<{ label: string; session: string; index: number; acc: number[][]; gyro: number[][] }> }
    expect(doc.format).toBe('curbcut-windows-v1')
    expect([doc.rate_hz, doc.window]).toEqual([50, 100])
    const w = doc.windows[0]!
    expect([w.label, w.session, w.index]).toEqual(['rough', 'abc123', 4])
    expect(w.acc.length).toBe(100)
    expect(w.acc[0]).toEqual([0.1235, 0, 9.81])
    expect(w.gyro[99]).toEqual([0, 0, 0])
  })
})
