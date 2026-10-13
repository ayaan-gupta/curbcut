import { haversine, type LonLat } from '../geo'
import type { Runner } from '../onnx'
import { WalkClassifier, type RoughnessClass } from '../roughness/classifier'
import { Resampler, type MotionSample, type TimedWindow } from './resample'

export interface Fix {
  t: number
  lon: number
  lat: number
  accuracy: number
}

export interface WindowResult {
  tStart: number
  tEnd: number
  label: RoughnessClass
  probs: [number, number, number]
  confidence: number
  /** Peak vertical acceleration in the window, m/s^2. */
  peak: number
  /** Where the phone was at the middle of the window, if GPS knew. */
  position: Fix | null
  raw: TimedWindow
}

export interface TaggedPoint {
  state: 'smooth' | 'rough' | 'bump'
  lon: number
  lat: number
  accuracy: number
  confidence: number
  value: number | null
  t: number
}

const MAX_FIX_AGE_MS = 5000
const MAX_ACCURACY_M = 25
const MIN_CONFIDENCE = 0.6
const MIN_SPACING_M = 6

/** Where the phone was at time t, by interpolating between the two fixes around it. */
export function positionAt(trace: Fix[], t: number): Fix | null {
  if (trace.length === 0) return null
  let after = trace.findIndex((f) => f.t >= t)
  if (after === -1) after = trace.length
  const a = trace[after - 1]
  const b = trace[after]
  if (a && b) {
    if (b.t - a.t > 2 * MAX_FIX_AGE_MS) return null
    const f = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t)
    return { t, lon: a.lon + f * (b.lon - a.lon), lat: a.lat + f * (b.lat - a.lat), accuracy: Math.max(a.accuracy, b.accuracy) }
  }
  const only = a ?? b!
  return Math.abs(only.t - t) <= MAX_FIX_AGE_MS ? { ...only, t } : null
}

/**
 * Which windows become reports.
 *
 * Bumps and rough stretches that the model is at least 60 percent sure of and
 * that GPS could place within 25 m. Rough ground is reported once every 6 m,
 * not once per window. A smooth window is only sent as counter-evidence when
 * it passes within 6 m of a surface issue already on the map, so a walk over
 * good pavement uploads nothing at all.
 */
export function selectTaggedPoints(windows: WindowResult[], knownSurfaceIssues: LonLat[] = []): TaggedPoint[] {
  const out: TaggedPoint[] = []
  const lastOf: Partial<Record<TaggedPoint['state'], LonLat>> = {}
  for (const w of windows) {
    const p = w.position
    if (!p || p.accuracy > MAX_ACCURACY_M || w.confidence < MIN_CONFIDENCE) continue
    const here: LonLat = [p.lon, p.lat]
    if (w.label === 'smooth' && !knownSurfaceIssues.some((k) => haversine(k, here) <= MIN_SPACING_M)) continue
    const last = lastOf[w.label]
    if (w.label !== 'bump' && last && haversine(last, here) < MIN_SPACING_M) continue
    lastOf[w.label] = here
    out.push({ state: w.label, lon: p.lon, lat: p.lat, accuracy: p.accuracy, confidence: w.confidence, value: w.label === 'bump' ? Math.round(w.peak * 10) / 10 : null, t: (w.tStart + w.tEnd) / 2 })
  }
  return out
}

/**
 * One recording session. Feed it motion samples and GPS fixes as they
 * arrive; it resamples, windows, classifies and keeps the trace. Nothing
 * here touches the network: the trace never leaves the phone, and only the
 * tagged points are offered for upload when the session ends.
 */
export class Recorder {
  readonly trace: Fix[] = []
  readonly windows: WindowResult[] = []
  private readonly resampler = new Resampler()
  private readonly classifier: WalkClassifier<TimedWindow>
  private chain: Promise<void> = Promise.resolve()

  constructor(runner: Runner, private readonly onWindow?: (w: WindowResult) => void) {
    this.classifier = new WalkClassifier<TimedWindow>(runner)
  }

  fix(f: Fix): void {
    if (f.accuracy <= 60) this.trace.push(f)
  }

  /** Samples must be handed over in time order. Classification is queued so results stay in order too. */
  motion(sample: MotionSample): void {
    const { windows, gap } = this.resampler.push(sample)
    if (gap) this.chain = this.chain.then(async () => this.emit(await this.classifier.breakWalk()))
    for (const w of windows) this.chain = this.chain.then(async () => this.emit(await this.classifier.push(w, w)))
  }

  async stop(): Promise<{ trace: Fix[]; windows: WindowResult[] }> {
    this.chain = this.chain.then(async () => this.emit(await this.classifier.flush()))
    await this.chain
    return { trace: this.trace, windows: this.windows }
  }

  private emit(result: Awaited<ReturnType<WalkClassifier<TimedWindow>['push']>>): void {
    if (!result) return
    const raw = result.meta
    const done: WindowResult = {
      tStart: raw.tStart,
      tEnd: raw.tEnd,
      label: result.label,
      probs: result.probs,
      confidence: result.confidence,
      peak: result.base[2]!,
      position: positionAt(this.trace, (raw.tStart + raw.tEnd) / 2),
      raw,
    }
    this.windows.push(done)
    this.onWindow?.(done)
  }
}
