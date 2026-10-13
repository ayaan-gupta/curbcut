import type { Runner } from '../onnx'
import { assemble, extractBase, type MotionWindow } from './features'

export const CLASSES = ['smooth', 'rough', 'bump'] as const
export type RoughnessClass = (typeof CLASSES)[number]

export interface Classified<T> {
  label: RoughnessClass
  probs: [number, number, number]
  confidence: number
  base: Float32Array
  meta: T
}

/** Runs the exported gradient-boosted model on one assembled feature vector. */
export async function predict(runner: Runner, features: Float32Array): Promise<{ label: RoughnessClass; probs: [number, number, number] }> {
  const out = await runner.run(features, [1, features.length])
  const probs = Array.from((out.probabilities ?? Object.values(out)[1])!.data as ArrayLike<number>, Number) as [number, number, number]
  let best = 0
  for (let i = 1; i < 3; i++) if (probs[i]! > probs[best]!) best = i
  return { label: CLASSES[best]!, probs }
}

/**
 * Classifies the windows of one walk in order.
 *
 * The model looks at the window before and the window after, and at the
 * walk's recent baseline, so each answer arrives one window (2 s) late:
 * `push` returns the result for the previous window, and `flush` returns the
 * last one when recording stops.
 */
export class WalkClassifier<T> {
  private history: Float32Array[] = []
  private prev: Float32Array | null = null
  private pending: { base: Float32Array; meta: T } | null = null

  constructor(private readonly runner: Runner) {}

  async push(window: MotionWindow, meta: T): Promise<Classified<T> | null> {
    const base = extractBase(window)
    const result = this.pending ? await this.finish(this.pending, base) : null
    this.pending = { base, meta }
    return result
  }

  async flush(): Promise<Classified<T> | null> {
    if (!this.pending) return null
    const result = await this.finish(this.pending, null)
    this.pending = null
    return result
  }

  /** A gap in the sensor stream ends the walk as far as context goes. */
  async breakWalk(): Promise<Classified<T> | null> {
    const result = await this.flush()
    this.prev = null
    return result
  }

  private async finish(item: { base: Float32Array; meta: T }, next: Float32Array | null): Promise<Classified<T>> {
    const features = assemble(item.base, this.history, this.prev, next)
    const { label, probs } = await predict(this.runner, features)
    this.history.push(item.base)
    if (this.history.length > 64) this.history.shift()
    this.prev = item.base
    return { label, probs, confidence: Math.max(...probs), base: item.base, meta: item.meta }
  }
}
