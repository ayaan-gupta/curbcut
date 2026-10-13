import { FS, WINDOW } from './roughness/features'
import type { RoughnessClass } from './roughness/classifier'
import type { TimedWindow } from './sensors/resample'

export interface LabelledWindow {
  label: RoughnessClass
  session: string
  index: number
  acc: number[][]
  gyro: number[][]
}

/**
 * The file ml/roughness/retrain.py reads. One entry per 2 s window, with the
 * label the person chose while recording, the session it came from and its
 * position in that session.
 */
export function exportWindows(windows: LabelledWindow[]): string {
  return JSON.stringify({ format: 'curbcut-windows-v1', rate_hz: FS, window: WINDOW, exported_at: new Date().toISOString(), windows })
}

const round = (v: number) => Math.round(v * 1e4) / 1e4

export function labelWindow(raw: TimedWindow, label: RoughnessClass, session: string, index: number): LabelledWindow {
  const acc: number[][] = []
  const gyro: number[][] = []
  for (let i = 0; i < WINDOW; i++) {
    acc.push([round(raw.ax[i]!), round(raw.ay[i]!), round(raw.az[i]!)])
    gyro.push([round(raw.gx[i]!), round(raw.gy[i]!), round(raw.gz[i]!)])
  }
  return { label, session, index, acc, gyro }
}
