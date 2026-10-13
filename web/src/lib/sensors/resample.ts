import { FS, WINDOW, type MotionWindow } from '../roughness/features'

export interface MotionSample {
  /** Milliseconds on any monotonic clock. */
  t: number
  /** Acceleration including gravity, m/s^2, device frame. */
  ax: number
  ay: number
  az: number
  /** Rotation rate, rad/s. */
  gx: number
  gy: number
  gz: number
}

export interface TimedWindow extends MotionWindow {
  tStart: number
  tEnd: number
}

const STEP_MS = 1000 / FS
const MAX_GAP_MS = 250

/**
 * Phones deliver motion events at whatever rate they like (60 Hz on iOS, 50
 * to 200 Hz on Android, with jitter). This resamples onto an exact 50 Hz grid
 * by linear interpolation and cuts consecutive, non-overlapping 2 s windows.
 * A gap longer than a quarter of a second throws away the partial window and
 * reports a break.
 */
export class Resampler {
  private last: MotionSample | null = null
  private nextT = 0
  private buf: number[][] = [[], [], [], [], [], []]
  private windowStart = 0

  push(sample: MotionSample): { windows: TimedWindow[]; gap: boolean } {
    const windows: TimedWindow[] = []
    let gap = false
    const last = this.last
    if (last === null || sample.t - last.t > MAX_GAP_MS || sample.t <= last.t) {
      gap = last !== null && sample.t - last.t > MAX_GAP_MS
      if (last !== null && sample.t <= last.t) return { windows, gap: false }
      this.buf = [[], [], [], [], [], []]
      this.nextT = sample.t
      this.windowStart = sample.t
      this.last = sample
      return { windows, gap }
    }
    while (this.nextT <= sample.t) {
      const f = (this.nextT - last.t) / (sample.t - last.t)
      const values = [
        last.ax + f * (sample.ax - last.ax), last.ay + f * (sample.ay - last.ay), last.az + f * (sample.az - last.az),
        last.gx + f * (sample.gx - last.gx), last.gy + f * (sample.gy - last.gy), last.gz + f * (sample.gz - last.gz),
      ]
      if (this.buf[0]!.length === 0) this.windowStart = this.nextT
      for (let i = 0; i < 6; i++) this.buf[i]!.push(values[i]!)
      if (this.buf[0]!.length === WINDOW) {
        const [ax, ay, az, gx, gy, gz] = this.buf as [number[], number[], number[], number[], number[], number[]]
        windows.push({ ax, ay, az, gx, gy, gz, tStart: this.windowStart, tEnd: this.windowStart + WINDOW * STEP_MS })
        this.buf = [[], [], [], [], [], []]
      }
      this.nextT += STEP_MS
    }
    this.last = sample
    return { windows, gap }
  }
}
