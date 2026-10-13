/**
 * Roughness features for one 2 second window of phone motion.
 *
 * This mirrors ml/roughness/features.py line for line. The shared fixture
 * ml/roughness/fixtures/feature_parity.json keeps the two in step: change
 * one side and the parity test on the other fails.
 *
 * Input: 100 samples at 50 Hz. Acceleration including gravity in the device
 * frame (m/s^2) and rotation rate (rad/s). The phone can be held any way up:
 * gravity is taken as the window mean and "vertical" is the part along it.
 */

export const FS = 50
export const WINDOW = 100
const EPS = 1e-9

export const FEATURE_NAMES = [
  'v_rms', 'v_p2p', 'v_maxabs', 'v_crest', 'v_kurtosis', 'h_rms', 'jerk_rms', 'jerk_max',
  'band_0p5_3', 'band_3_8', 'band_8_15', 'band_15_25', 'hf_ratio', 'dom_freq', 'centroid',
  'gyro_rms', 'gyro_max', 'zcr', 'burst_ratio', 'top5_energy', 'step_regularity',
] as const
export const RELATIVE_NAMES = ['rel_v_rms', 'rel_jerk_rms', 'rel_hf'] as const
export const CONTEXT_SOURCE = ['v_rms', 'v_maxabs', 'jerk_rms', 'burst_ratio', 'band_8_15', 'band_15_25'] as const
export const ALL_FEATURE_NAMES: string[] = [
  ...FEATURE_NAMES,
  ...RELATIVE_NAMES,
  ...CONTEXT_SOURCE.map((n) => `prev_${n}`),
  ...CONTEXT_SOURCE.map((n) => `next_${n}`),
]
export const BASELINE_WINDOWS = 20
export const MIN_BASELINE = 3

const idx = (name: (typeof FEATURE_NAMES)[number]) => FEATURE_NAMES.indexOf(name)
const V_RMS = idx('v_rms')
const JERK_RMS = idx('jerk_rms')
const B3 = idx('band_8_15')
const B4 = idx('band_15_25')
const CONTEXT_IDX = CONTEXT_SOURCE.map((n) => idx(n))

/** Inclusive DFT bin ranges. With N=100 and fs=50, bin k is 0.5*k Hz. */
const BANDS: Array<[number, number]> = [[1, 5], [6, 15], [16, 29], [30, 50]]
const HANN = Float64Array.from({ length: WINDOW }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / WINDOW))
const HANN_POWER = HANN.reduce((s, w) => s + w * w, 0)
// Precomputed cos and sin tables for a direct 100 point DFT. 5,000 multiplies per window is nothing.
const COS = Float64Array.from({ length: WINDOW }, (_, i) => Math.cos((2 * Math.PI * i) / WINDOW))
const SIN = Float64Array.from({ length: WINDOW }, (_, i) => Math.sin((2 * Math.PI * i) / WINDOW))

/** A window as six parallel arrays of length 100. */
export interface MotionWindow {
  ax: ArrayLike<number>
  ay: ArrayLike<number>
  az: ArrayLike<number>
  gx: ArrayLike<number>
  gy: ArrayLike<number>
  gz: ArrayLike<number>
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
}

/** The 21 base features of one window. */
export function extractBase(w: MotionWindow): Float32Array {
  const n = WINDOW
  if (w.ax.length !== n || w.gx.length !== n) throw new Error(`expected ${n} samples, got ${w.ax.length}`)
  let mx = 0, my = 0, mz = 0
  for (let i = 0; i < n; i++) { mx += w.ax[i]!; my += w.ay[i]!; mz += w.az[i]! }
  mx /= n; my /= n; mz /= n
  const norm = Math.hypot(mx, my, mz)
  const ux = mx / (norm + EPS), uy = my / (norm + EPS), uz = mz / (norm + EPS)

  const v = new Float64Array(n)
  let energy = 0, hSum = 0, vMax = -Infinity, vMin = Infinity, vAbs = 0, m4 = 0
  for (let i = 0; i < n; i++) {
    const along = w.ax[i]! * ux + w.ay[i]! * uy + w.az[i]! * uz
    const vi = along - norm
    v[i] = vi
    const hx = w.ax[i]! - along * ux, hy = w.ay[i]! - along * uy, hz = w.az[i]! - along * uz
    hSum += hx * hx + hy * hy + hz * hz
    energy += vi * vi
    m4 += vi * vi * vi * vi
    if (vi > vMax) vMax = vi
    if (vi < vMin) vMin = vi
    if (Math.abs(vi) > vAbs) vAbs = Math.abs(vi)
  }
  const vRms = Math.sqrt(energy / n)
  const m2 = energy / n
  const vKurt = m4 / n / (m2 * m2 + EPS)
  const hRms = Math.sqrt(hSum / n)

  let jerkSum = 0, jerkMax = 0
  for (let i = 1; i < n; i++) {
    const j = (v[i]! - v[i - 1]!) * FS
    jerkSum += j * j
    if (Math.abs(j) > jerkMax) jerkMax = Math.abs(j)
  }
  const jerkRms = Math.sqrt(jerkSum / (n - 1))

  // Hann-windowed power per bin, bins 1 to N/2.
  const power = new Float64Array(n / 2 + 1)
  for (let k = 1; k <= n / 2; k++) {
    let re = 0, im = 0
    for (let i = 0; i < n; i++) {
      const x = v[i]! * HANN[i]!
      const a = (k * i) % n
      re += x * COS[a]!
      im -= x * SIN[a]!
    }
    power[k] = (re * re + im * im) / HANN_POWER
  }
  const bands = BANDS.map(([lo, hi]) => { let s = 0; for (let k = lo; k <= hi; k++) s += power[k]!; return s })
  const total = bands.reduce((s, b) => s + b, 0)
  const hfRatio = (bands[2]! + bands[3]!) / (total + EPS)
  let domK = 1, pSum = 0, fpSum = 0
  for (let k = 1; k <= n / 2; k++) {
    if (power[k]! > power[domK]!) domK = k
    pSum += power[k]!
    fpSum += ((k * FS) / n) * power[k]!
  }
  const domFreq = (domK * FS) / n
  const centroid = fpSum / (pSum + EPS)

  let gmx = 0, gmy = 0, gmz = 0
  for (let i = 0; i < n; i++) { gmx += w.gx[i]!; gmy += w.gy[i]!; gmz += w.gz[i]! }
  gmx /= n; gmy /= n; gmz /= n
  let gSum = 0, gMax = 0
  for (let i = 0; i < n; i++) {
    const m = (w.gx[i]! - gmx) ** 2 + (w.gy[i]! - gmy) ** 2 + (w.gz[i]! - gmz) ** 2
    gSum += m
    if (m > gMax) gMax = m
  }
  const gyroRms = Math.sqrt(gSum / n)
  const gyroMax = Math.sqrt(gMax)

  let crossings = 0
  for (let i = 1; i < n; i++) if (v[i]! >= 0 !== v[i - 1]! >= 0) crossings++
  const zcr = crossings / n

  // Ten consecutive 0.2 s blocks: how far the loudest stands out from the typical one.
  const blocks: number[] = []
  for (let b = 0; b < 10; b++) {
    let s = 0
    for (let i = b * 10; i < b * 10 + 10; i++) s += v[i]! * v[i]!
    blocks.push(Math.sqrt(s / 10))
  }
  const sortedBlocks = [...blocks].sort((a, b) => a - b)
  const burstRatio = sortedBlocks[9]! / (sortedBlocks[4]! + 0.05)

  const squares = Array.from(v, (x) => x * x).sort((a, b) => a - b)
  let top5 = 0
  for (let i = n - 5; i < n; i++) top5 += squares[i]!
  const top5Energy = top5 / (energy + EPS)

  // Gait periodicity: best normalised autocorrelation for lags of 0.3 to 0.8 s.
  let best = -1
  for (let lag = 15; lag <= 40; lag++) {
    let num = 0, a = 0, b = 0
    for (let i = 0; i < n - lag; i++) {
      num += v[i]! * v[i + lag]!
      a += v[i]! * v[i]!
      b += v[i + lag]! * v[i + lag]!
    }
    const r = num / (Math.sqrt(a * b) + EPS)
    if (r > best) best = r
  }

  return Float32Array.from([
    vRms, vMax - vMin, vAbs, vAbs / (vRms + EPS), vKurt, hRms, jerkRms, jerkMax,
    Math.log10(bands[0]! + 1e-6), Math.log10(bands[1]! + 1e-6), Math.log10(bands[2]! + 1e-6), Math.log10(bands[3]! + 1e-6),
    hfRatio, domFreq, centroid, gyroRms, gyroMax, zcr, burstRatio, top5Energy, best,
  ])
}

/** log10 of the power above 8 Hz, rebuilt from the two logged band features. */
export function hfLogPower(base: ArrayLike<number>): number {
  return Math.log10(10 ** base[B3]! + 10 ** base[B4]!)
}

/** The three features relative to this walk's recent past (oldest first). */
export function relative(base: ArrayLike<number>, history: ArrayLike<number>[]): [number, number, number] {
  const recent = history.slice(-BASELINE_WINDOWS)
  if (recent.length < MIN_BASELINE) return [1, 1, 0]
  const medV = median(recent.map((h) => h[V_RMS]!))
  const medJ = median(recent.map((h) => h[JERK_RMS]!))
  const medHf = median(recent.map((h) => hfLogPower(h)))
  return [base[V_RMS]! / (medV + 1e-6), base[JERK_RMS]! / (medJ + 1e-6), hfLogPower(base) - medHf]
}

/** The full model input for one window. At the ends of a walk a missing neighbour is replaced by the window itself. */
export function assemble(base: Float32Array, history: ArrayLike<number>[], prev: ArrayLike<number> | null, next: ArrayLike<number> | null): Float32Array {
  const before = prev ?? base
  const after = next ?? base
  return Float32Array.from([...base, ...relative(base, history), ...CONTEXT_IDX.map((i) => before[i]!), ...CONTEXT_IDX.map((i) => after[i]!)])
}
