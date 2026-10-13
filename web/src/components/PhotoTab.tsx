import { useEffect, useRef, useState } from 'react'
import art from 'virtual:art'
import { api, type SubmitResult } from '../lib/api'
import type { LonLat } from '../lib/geo'
import { webRunner } from '../lib/onnx'
import { blurRegions, type Region } from '../lib/photo/blur'
import { drawPixels, fileToPixels, pixelsToJpeg } from '../lib/photo/browser'
import { classifyPhoto, PHOTO_CLASSES, type PhotoClass, type PhotoVerdict } from '../lib/photo/classify'
import { detectFaces } from '../lib/photo/faces'
import type { Pixels } from '../lib/photo/pixels'
import { iconUrl } from './MapView'

const CLASS_LABEL: Record<PhotoClass, string> = { present: 'Curb cut present', missing: 'No curb cut', damaged: 'Damaged', blocked: 'Blocked' }
const STEPS = [
  ['onboarding/step-1-stand-back', 'Stand two or three steps back.'],
  ['onboarding/step-2-frame-the-edge', 'Frame the whole ramp, where the sidewalk meets the road.'],
  ['onboarding/step-3-faces-and-plates', 'Faces are covered on your phone before anything is sent.'],
] as const

interface Props {
  picked: LonLat | null
  onNeedPoint: () => void
  onSubmitted: () => void
}

export function PhotoTab({ picked, onNeedPoint, onSubmitted }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const photo = useRef<Pixels | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [verdict, setVerdict] = useState<PhotoVerdict | null>(null)
  const [label, setLabel] = useState<PhotoClass>('present')
  const [faces, setFaces] = useState(0)
  const [manual, setManual] = useState(0)
  const [sample, setSample] = useState(false)
  const [result, setResult] = useState<SubmitResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [gps, setGps] = useState<{ at: LonLat; accuracy: number } | null>(null)
  const drag = useRef<{ x: number; y: number } | null>(null)

  const load = async (file: Blob, isSample: boolean) => {
    setError(null)
    setResult(null)
    setVerdict(null)
    setManual(0)
    setSample(isSample)
    try {
      setBusy('Reading the photo')
      const pixels = await fileToPixels(file)
      // Classify first, on the untouched pixels, then cover faces before the photo is shown or kept.
      setBusy('Classifying on this device')
      const v = await classifyPhoto(await webRunner('/models/curb-photo.onnx'), pixels)
      setBusy('Looking for faces to blur')
      const found = await detectFaces(await webRunner('/models/face-ultraface-320.onnx'), pixels)
      blurRegions(pixels, found)
      photo.current = pixels
      setFaces(found.length)
      setVerdict(v)
      setLabel(v.label)
      if (canvas.current) drawPixels(canvas.current, pixels)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  useEffect(() => {
    if (photo.current && canvas.current) drawPixels(canvas.current, photo.current)
  }, [verdict])

  const toPhoto = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return { x: ((e.clientX - rect.left) / rect.width) * e.currentTarget.width, y: ((e.clientY - rect.top) / rect.height) * e.currentTarget.height }
  }
  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const from = drag.current
    drag.current = null
    if (!from || !photo.current) return
    const to = toPhoto(e)
    const region: Region = { x: Math.min(from.x, to.x), y: Math.min(from.y, to.y), width: Math.abs(to.x - from.x), height: Math.abs(to.y - from.y) }
    if (region.width < 8 || region.height < 8) return
    blurRegions(photo.current, [region], 0)
    drawPixels(e.currentTarget, photo.current)
    setManual((n) => n + 1)
  }

  const locate = () =>
    navigator.geolocation?.getCurrentPosition(
      (p) => setGps({ at: [p.coords.longitude, p.coords.latitude], accuracy: p.coords.accuracy }),
      () => setError('Location was not available. Tap the map to place the photo instead.'),
      { enableHighAccuracy: true, timeout: 10000 },
    )

  const where = picked ? { at: picked, accuracy: 6 } : gps
  const submit = async () => {
    if (!photo.current || !verdict || !where) return
    setBusy('Sending')
    setError(null)
    try {
      const blob = await pixelsToJpeg(photo.current)
      const sent = await api.submit(
        {
          category: 'curb',
          state: label,
          lon: where.at[0],
          lat: where.at[1],
          accuracy_m: where.accuracy,
          source: 'photo',
          // If the person corrected the model, their choice is trusted about as much as a manual report.
          model_confidence: label === verdict.label ? verdict.confidence : Math.max(0.75, verdict.probs[label]),
          simulated: sample,
          faces_blurred: faces,
          regions_blurred: manual,
        },
        blob,
      )
      setResult(sent)
      onSubmitted()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const pickSample = async (name: string) => load(await (await fetch(`/samples/${name}`)).blob(), true)

  return (
    <div className="stack">
      {!verdict && (
        <>
          <div className="steps">
            {STEPS.map(([name, caption]) => (
              <figure key={name}>
                <img src={art[name]?.url} alt="" />
                <figcaption>{caption}</figcaption>
              </figure>
            ))}
          </div>
          <label className="primary file">
            Take or choose a photo
            <input type="file" accept="image/*" capture="environment" onChange={(e) => e.target.files?.[0] && void load(e.target.files[0], false)} data-testid="photo-input" />
          </label>
          <div className="inline">
            <span className="muted">No camera here?</span>
            <button className="pill ghost" onClick={() => void pickSample('curb-cut.jpg')} data-testid="sample-curb">Sample curb</button>
            <button className="pill ghost" onClick={() => void pickSample('faces.jpg')} data-testid="sample-faces">Sample with faces</button>
          </div>
        </>
      )}
      {busy && <div className="card note" data-testid="photo-busy">{busy}…</div>}
      {error && <div className="card warn">{error}</div>}

      <div className={verdict ? 'photo-wrap' : 'photo-wrap hidden'}>
        <canvas ref={canvas} className="photo" data-testid="photo-canvas" onPointerDown={(e) => { e.preventDefault(); drag.current = toPhoto(e); e.currentTarget.setPointerCapture(e.pointerId) }} onPointerUp={endDrag} />
      </div>

      {verdict && (
        <>
          <p className="muted" data-testid="blur-summary">
            {faces === 0 ? 'No faces found.' : `${faces} face${faces === 1 ? '' : 's'} covered on this device.`} {manual > 0 ? `${manual} area${manual === 1 ? '' : 's'} covered by hand. ` : ''}
            Licence plates are not found automatically: drag across one to cover it.
          </p>
          <div className="card" data-testid="photo-verdict" data-label={verdict.label}>
            {PHOTO_CLASSES.map((c) => (
              <button key={c} className={`prob ${label === c ? 'on' : ''}`} onClick={() => setLabel(c)} data-testid={`label-${c}`}>
                <img src={iconUrl(`curb_${c}`)} width={24} height={24} alt="" />
                <span>{CLASS_LABEL[c]}</span>
                <span className="bar"><i style={{ width: `${Math.round(verdict.probs[c] * 100)}%` }} /></span>
                <b>{Math.round(verdict.probs[c] * 100)}%</b>
              </button>
            ))}
            <p className="fine">The model's guess is selected. It is wrong about three times in ten on real photos, so tap the right answer if it missed.</p>
          </div>
          <div className="rows">
            <div className="row">
              <span className="grow">{where ? `Placed at ${where.at[1].toFixed(5)}, ${where.at[0].toFixed(5)}` : 'Where is this curb?'}</span>
              <button className="pill ghost" onClick={locate}>My location</button>
              <button className="pill ghost" onClick={onNeedPoint} data-testid="photo-pick">Tap the map</button>
            </div>
          </div>
          {sample && <span className="badge">Sample photo: stored as synthetic</span>}
          {!result && (
            <div className="inline">
              <button className="primary" disabled={!where || busy !== null} onClick={() => void submit()} data-testid="photo-submit">Send report</button>
              <button className="pill ghost" onClick={() => { setVerdict(null); photo.current = null }}>Start over</button>
            </div>
          )}
          {result && (
            <div className="card" data-testid="photo-result">
              <p>{result.message}</p>
              <button className="pill" onClick={() => { setVerdict(null); setResult(null); photo.current = null }}>Another photo</button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
