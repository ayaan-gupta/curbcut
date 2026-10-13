import { useEffect, useRef, useState } from 'react'
import { api, download, type Features, type ReportIn } from '../lib/api'
import { exportWindows, labelWindow, type LabelledWindow } from '../lib/collect'
import { formatDistance, travelled, type LonLat } from '../lib/geo'
import { webRunner } from '../lib/onnx'
import type { RoughnessClass } from '../lib/roughness/classifier'
import { Recorder, selectTaggedPoints, type TaggedPoint, type WindowResult } from '../lib/sensors/recorder'
import { deviceSensorsAvailable, deviceSource, simulatedSource, type SensorSource } from '../lib/sensors/sources'
import type { TaggedDot } from './MapView'

interface Props {
  features: Features | null
  onTrace: (trace: LonLat[], tagged: TaggedDot[]) => void
  onUploaded: () => void
}

type Phase = 'idle' | 'starting' | 'recording' | 'review' | 'uploading' | 'done'
const CLASS_LABEL: Record<RoughnessClass, string> = { smooth: 'Smooth', rough: 'Rough', bump: 'Bump' }
const COLLECT_KEY = 'curbcut.labelled'

function Spark({ values }: { values: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)
    ctx.strokeStyle = '#111'
    ctx.lineWidth = 2
    ctx.beginPath()
    const scale = Math.max(4, ...values.map((v) => Math.abs(v)))
    values.forEach((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * width
      const y = height / 2 - (v / scale) * (height / 2 - 3)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
  }, [values])
  return <canvas ref={ref} width={640} height={96} className="spark" aria-label="Vertical acceleration, last few seconds" />
}

/** Vertical acceleration of a window, for the live trace: the part of each sample along the window's mean. */
function vertical(w: WindowResult): number[] {
  const n = w.raw.ax.length
  let mx = 0, my = 0, mz = 0
  for (let i = 0; i < n; i++) { mx += w.raw.ax[i]!; my += w.raw.ay[i]!; mz += w.raw.az[i]! }
  const norm = Math.hypot(mx, my, mz) / n
  const out: number[] = []
  for (let i = 0; i < n; i += 2) out.push((w.raw.ax[i]! * mx + w.raw.ay[i]! * my + w.raw.az[i]! * mz) / (norm * n) - norm)
  return out
}

export function RecordTab({ features, onTrace, onUploaded }: Props) {
  const hasSensors = deviceSensorsAvailable()
  const [useSim, setUseSim] = useState(!hasSensors)
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [windows, setWindows] = useState<WindowResult[]>([])
  const [points, setPoints] = useState<TaggedPoint[]>([])
  const [message, setMessage] = useState<string | null>(null)
  const [collect, setCollect] = useState(false)
  const [truth, setTruth] = useState<RoughnessClass>('smooth')
  const [labelled, setLabelled] = useState<LabelledWindow[]>(() => {
    try { return JSON.parse(localStorage.getItem(COLLECT_KEY) ?? '[]') as LabelledWindow[] } catch { return [] }
  })
  const [slope, setSlope] = useState('')
  const recorder = useRef<Recorder | null>(null)
  const source = useRef<SensorSource | null>(null)
  const session = useRef('')
  const state = useRef({ collect, truth, bumpNext: false })
  state.current.collect = collect
  state.current.truth = truth
  const known = useRef<LonLat[]>([])
  known.current = (features?.features ?? []).filter((f) => f.properties.category === 'surface').map((f) => f.geometry.coordinates as LonLat)

  const publish = (rec: Recorder) => {
    const tagged = selectTaggedPoints(rec.windows, known.current)
    onTrace(rec.trace.map((f) => [f.lon, f.lat] as LonLat), tagged.filter((t) => t.state !== 'smooth').map((t) => ({ lon: t.lon, lat: t.lat, state: t.state })))
    return tagged
  }

  const stop = async () => {
    source.current?.stop()
    const rec = recorder.current
    if (!rec) return
    await rec.stop()
    setWindows([...rec.windows])
    setPoints(publish(rec))
    setPhase('review')
  }

  const start = async () => {
    setError(null)
    setMessage(null)
    setWindows([])
    setPoints([])
    setPhase('starting')
    try {
      const runner = await webRunner('/models/roughness.onnx')
      session.current = Math.random().toString(36).slice(2, 8)
      let index = 0
      const rec = new Recorder(runner, (w) => {
        setWindows((prev) => [...prev, w])
        publish(rec)
        if (state.current.collect) {
          const label = state.current.bumpNext ? 'bump' : state.current.truth
          state.current.bumpNext = false
          setLabelled((prev) => {
            const next = [...prev, labelWindow(w.raw, label, session.current, index++)]
            try { localStorage.setItem(COLLECT_KEY, JSON.stringify(next.slice(-400))) } catch { /* storage full: keep in memory */ }
            return next
          })
        }
      })
      recorder.current = rec
      const src = useSim ? simulatedSource() : deviceSource()
      source.current = src
      await src.start((s) => rec.motion(s), (f) => rec.fix(f), () => void stop())
      setPhase('recording')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('idle')
    }
  }

  useEffect(() => () => source.current?.stop(), [])

  const upload = async () => {
    setPhase('uploading')
    const now = Date.now()
    const last = recorder.current?.windows.at(-1)?.tEnd ?? 0
    const reports: ReportIn[] = points.map((p) => ({
      category: 'surface',
      state: p.state,
      lon: p.lon,
      lat: p.lat,
      accuracy_m: p.accuracy,
      source: 'sensor',
      model_confidence: p.confidence,
      value: p.value,
      // The simulated feed plays faster than real time, so its points are spread over the last minute instead.
      recorded_at: new Date(now - (useSim ? Math.max(0, last - p.t) / 2 : Math.max(0, last - p.t))).toISOString(),
      simulated: useSim,
    }))
    try {
      const result = await api.submitBatch(reports)
      const merged = result.results.filter((r) => r.merged).length
      const live = result.results.filter((r) => r.feature?.published).length
      setMessage(`${result.accepted} point${result.accepted === 1 ? '' : 's'} sent. ${merged} merged with earlier reports, ${live} now on the open map.${result.rate_limited ? ' ' + result.rate_limited : ''}`)
      setPhase('done')
      onUploaded()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('review')
    }
  }

  const discard = () => {
    recorder.current = null
    setWindows([])
    setPoints([])
    onTrace([], [])
    setPhase('idle')
  }

  const reportSlope = async () => {
    const value = Number(slope)
    if (!(value > 0)) return
    setError(null)
    navigator.geolocation?.getCurrentPosition(
      async (pos) => {
        try {
          const result = await api.submit({ category: 'slope', state: value >= 5 ? 'steep' : 'gentle', lon: pos.coords.longitude, lat: pos.coords.latitude, accuracy_m: pos.coords.accuracy, source: 'manual', model_confidence: 0.9, value })
          setMessage(result.message)
          onUploaded()
        } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
      },
      () => setError('Location is needed to place a slope report.'),
      { enableHighAccuracy: true, timeout: 10000 },
    )
  }

  const measureSlope = () => {
    const samples: number[] = []
    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity
      if (a?.x == null || a.y == null || a.z == null) return
      // Lying flat on the ground, the screen normal tilts away from gravity by the slope angle.
      const tilt = Math.acos(Math.min(1, Math.abs(a.z) / Math.hypot(a.x, a.y, a.z)))
      samples.push(Math.tan(tilt) * 100)
    }
    window.addEventListener('devicemotion', onMotion)
    window.setTimeout(() => {
      window.removeEventListener('devicemotion', onMotion)
      if (samples.length) setSlope((samples.reduce((s, v) => s + v, 0) / samples.length).toFixed(1))
      else setError('No motion sensor answered. Type the slope instead.')
    }, 1500)
  }

  const last = windows.at(-1)
  const counts = { smooth: 0, rough: 0, bump: 0 }
  for (const w of windows) counts[w.label]++
  const trace = recorder.current?.trace ?? []
  const distance = travelled(trace.map((f) => [f.lon, f.lat] as LonLat))
  const recent = windows.slice(-3).flatMap(vertical)
  const toSend = points.length

  return (
    <div className="stack">
      {phase === 'idle' || phase === 'starting' ? (
        <>
          <p>
            Walk or roll with the app open. The phone feels the ground through its motion sensors and tags rough stretches and bumps on your GPS trace.
          </p>
          <ul className="plain">
            <li>Sensors are read only between Start and Stop, and recording ends if you leave the app.</li>
            <li>Your trace stays on this phone. Only the tagged points are offered for upload, and you choose.</li>
          </ul>
          {!hasSensors && <div className="card note">This device has no motion sensors, so recording replays a simulated roll down a real sidewalk. Anything sent from it is stored as synthetic.</div>}
          {hasSensors && (
            <label className="row toggle">
              <input type="checkbox" checked={useSim} onChange={(e) => setUseSim(e.target.checked)} />
              <span>Use the simulated feed instead of this phone's sensors</span>
            </label>
          )}
          <button className="primary" onClick={start} disabled={phase === 'starting'} data-testid="record-start">
            {phase === 'starting' ? 'Starting…' : useSim ? 'Start simulated walk' : 'Start recording'}
          </button>
        </>
      ) : null}

      {phase === 'recording' && (
        <>
          {useSim && <span className="badge">Simulated sensor feed, replayed 8 times faster</span>}
          <div className={`live live-${last?.label ?? 'smooth'}`} data-testid="live-class" data-class={last?.label ?? ''}>
            <div className="big">{last ? CLASS_LABEL[last.label] : 'Listening'}</div>
            <div className="muted">{last ? `${Math.round(last.confidence * 100)}% sure, 2 s ago` : 'First answer after 4 seconds'}</div>
          </div>
          <Spark values={recent} />
          <div className="stats">
            <div><b>{formatDistance(distance)}</b><span>travelled</span></div>
            <div><b>{counts.rough}</b><span>rough windows</span></div>
            <div><b>{counts.bump}</b><span>bumps</span></div>
          </div>
          {collect && (
            <div className="card">
              <div className="muted">What are you on right now? Each 2 s window is saved with this label.</div>
              <div className="segmented">
                {(['smooth', 'rough'] as RoughnessClass[]).map((c) => (
                  <button key={c} className={truth === c ? 'on' : ''} onClick={() => setTruth(c)}>{CLASS_LABEL[c]}</button>
                ))}
                <button onClick={() => (state.current.bumpNext = true)}>Bump now</button>
              </div>
            </div>
          )}
          <button className="primary stop" onClick={() => void stop()} data-testid="record-stop">Stop</button>
        </>
      )}

      {(phase === 'review' || phase === 'uploading' || phase === 'done') && (
        <div className="card" data-testid="record-summary">
          <div className="big small">{formatDistance(distance)} recorded</div>
          <p className="muted">
            {windows.length} windows: {counts.smooth} smooth, {counts.rough} rough, {counts.bump} with a bump.
          </p>
          {phase !== 'done' && (
            <>
              <p>
                {toSend === 0 ? 'Nothing worth reporting, so nothing to send.' : `${toSend} tagged point${toSend === 1 ? '' : 's'} can be sent. The trace itself is not uploaded.`}
              </p>
              <div className="inline">
                {toSend > 0 && (
                  <button className="primary" onClick={upload} disabled={phase === 'uploading'} data-testid="record-upload">
                    {phase === 'uploading' ? 'Sending…' : `Send ${toSend} point${toSend === 1 ? '' : 's'}`}
                  </button>
                )}
                <button className="pill ghost" onClick={discard}>Discard</button>
              </div>
            </>
          )}
          {message && <p data-testid="record-message">{message}</p>}
          {phase === 'done' && <button className="pill" onClick={discard}>New recording</button>}
        </div>
      )}
      {error && <div className="card warn">{error}</div>}
      {phase !== 'recording' && message && phase !== 'done' && <div className="card">{message}</div>}

      <details className="card">
        <summary>Measure a slope</summary>
        <p className="muted">Lay the phone flat on the ramp, screen up, and tap Measure. Or type a number from a level.</p>
        <div className="inline">
          <input className="field" inputMode="decimal" placeholder="Slope in %" value={slope} onChange={(e) => setSlope(e.target.value)} aria-label="Slope in percent" />
          <button className="pill ghost" onClick={measureSlope}>Measure</button>
          <button className="pill" onClick={() => void reportSlope()} disabled={!(Number(slope) > 0)}>Report here</button>
        </div>
      </details>

      <details className="card" open={collect}>
        <summary>Label training data</summary>
        <p className="muted">
          The roughness model was trained on simulated walks. To replace them with real ones, switch this on, record, and say what you are rolling over. Export the file and run <code>ml/roughness/retrain.py</code> on it.
        </p>
        <label className="row toggle">
          <input type="checkbox" checked={collect} onChange={(e) => setCollect(e.target.checked)} data-testid="collect-toggle" />
          <span>Save labelled windows while recording</span>
        </label>
        <div className="inline">
          <span className="muted" data-testid="collect-count">{labelled.length} windows saved on this device</span>
          <button className="pill ghost" disabled={!labelled.length} onClick={() => download('curbcut-windows.json', exportWindows(labelled), 'application/json')} data-testid="collect-export">Export</button>
          <button className="pill ghost" disabled={!labelled.length} onClick={() => { setLabelled([]); localStorage.removeItem(COLLECT_KEY) }}>Clear</button>
        </div>
      </details>
    </div>
  )
}
