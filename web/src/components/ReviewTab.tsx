import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, download, setReviewerToken, type Candidate, type Meta, type ReviewQueue } from '../lib/api'
import { KIND_LABEL } from './MapTab'

interface Props {
  meta: Meta
  onChanged: () => void
}

const FLAG: Record<string, string> = {
  implausible_speed: 'moved too fast between reports',
  far_from_any_path: 'far from any footway',
  contradicts_established_feature: 'contradicts a well-supported feature',
}

export function ReviewTab({ meta, onChanged }: Props) {
  const [queue, setQueue] = useState<ReviewQueue | null>(null)
  const [cands, setCands] = useState<Candidate[]>([])
  const [unmatched, setUnmatched] = useState(0)
  const [needToken, setNeedToken] = useState(false)
  const [token, setToken] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [allowSynthetic, setAllowSynthetic] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [q, c] = await Promise.all([api.review(), api.candidates()])
      setQueue(q)
      setCands(c.candidates)
      setUnmatched(c.unmatched.length)
      setNeedToken(false)
      setError(null)
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setNeedToken(true)
      else setError(e instanceof Error ? e.message : String(e))
    }
  }, [])
  useEffect(() => void refresh(), [refresh])

  const act = async (work: () => Promise<unknown>) => {
    try {
      await work()
      await refresh()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  if (needToken)
    return (
      <div className="stack">
        <p>Reviewing needs the reviewer token this server was started with.</p>
        <div className="inline">
          <input className="field" type="password" placeholder="Reviewer token" value={token} onChange={(e) => setToken(e.target.value)} aria-label="Reviewer token" />
          <button className="pill" onClick={() => { setReviewerToken(token); void refresh() }}>Unlock</button>
        </div>
      </div>
    )

  const approved = cands.filter((c) => c.decision === 'approved' && c.kind !== 'confirm')
  const syntheticApproved = approved.filter((c) => c.synthetic).length
  const downloadDraft = async () => {
    const xml = await api.osmChange(allowSynthetic)
    const count = (xml.match(/<(node|way) /g) ?? []).length
    setNote(count === 0 ? 'The draft is empty: nothing approved yet, or everything approved rests on synthetic demo data.' : `Draft with ${count} element${count === 1 ? '' : 's'} saved.`)
    download(xml.includes('DO NOT UPLOAD') ? 'curbcut-draft-SYNTHETIC-do-not-upload.osc' : 'curbcut-draft.osc', xml, 'application/xml')
  }

  return (
    <div className="stack">
      {error && <div className="card warn">{error}</div>}
      <h3>Held for review</h3>
      {queue?.held_reports.length === 0 && <p className="muted">Nothing is waiting.</p>}
      {queue?.held_reports.map((r) => (
        <div className="card" key={r.id} data-testid="held-report">
          <div className="feature-head">
            <div>
              <div className="big small">{KIND_LABEL[`${r.category}_${r.state}`] ?? r.state}</div>
              <div className="muted">Held because it {r.flags.map((f) => FLAG[f] ?? f.replace(/_/g, ' ')).join(' and ')}.</div>
            </div>
          </div>
          <div className="inline">
            <button className="pill" onClick={() => void act(() => api.reviewReport(r.id, 'accept'))} data-testid="held-accept">Count it</button>
            <button className="pill ghost" onClick={() => void act(() => api.reviewReport(r.id, 'reject'))}>Reject</button>
            {r.synthetic && <span className="badge">Synthetic</span>}
          </div>
        </div>
      ))}

      <h3>Reporters disagree</h3>
      {queue?.conflicts.length === 0 && <p className="muted">No conflicts.</p>}
      {queue?.conflicts.map((f) => (
        <div className="card" key={f.id}>
          <div className="big small">{KIND_LABEL[f.kind] ?? f.kind}?</div>
          <p className="muted">{f.reporters} reporters, {Math.round(f.agreement * 100)}% agree. Confidence {f.confidence.toFixed(2)}, below the {meta.publish_threshold.toFixed(2)} needed.</p>
          <div className="inline">
            <button className="pill" onClick={() => void act(() => api.reviewFeature(f.id, 'approve'))}>Confirm and publish</button>
            <button className="pill ghost" onClick={() => void act(() => api.reviewFeature(f.id, 'reject'))}>Dismiss</button>
          </div>
        </div>
      ))}
      {queue && queue.below_threshold.length > 0 && <p className="muted">{queue.below_threshold.length} more report{queue.below_threshold.length === 1 ? ' is' : 's are'} waiting for a second reporter. One report alone is never published.</p>}

      <h3>OpenStreetMap draft</h3>
      <p className="muted">
        Well-supported features become proposed tag edits. Nothing here is sent to OpenStreetMap. You approve edits one by one, download an OsmChange file, open it in JOSM, check it, and upload it under your own account if you agree.
      </p>
      {cands.map((c) => (
        <div className="card" key={c.id} data-testid="candidate" data-kind={c.kind}>
          <div className="tagline">
            <a href={`https://www.openstreetmap.org/${c.element}/${c.osm_id}`} target="_blank" rel="noreferrer">{c.element} {c.osm_id}</a>
            <span className={`kind kind-${c.kind}`}>{c.kind === 'confirm' ? 'already tagged' : c.kind === 'change' ? 'changes a tag' : 'adds a tag'}</span>
          </div>
          <code className="tag">
            {c.key}={c.current ?? '(none)'} <span aria-hidden>→</span> <b>{c.key}={c.value}</b>
          </code>
          <p className="muted">{c.reporters} reporters, confidence {c.confidence.toFixed(2)}, {c.distance_m} m from the element. {c.note}</p>
          {c.kind !== 'confirm' && (
            <div className="inline">
              {c.key === 'incline' && c.decision !== 'approved' ? (
                <>
                  <button className="pill" onClick={() => void act(() => api.decide(c.id, 'approved', 1))}>Approve, uphill along the way</button>
                  <button className="pill" onClick={() => void act(() => api.decide(c.id, 'approved', -1))}>Approve, downhill</button>
                </>
              ) : (
                <button className={`pill ${c.decision === 'approved' ? 'on' : ''}`} onClick={() => void act(() => api.decide(c.id, c.decision === 'approved' ? 'pending' : 'approved'))} data-testid="candidate-approve">
                  {c.decision === 'approved' ? 'Approved' : 'Approve'}
                </button>
              )}
              <button className={`pill ghost ${c.decision === 'skipped' ? 'on' : ''}`} onClick={() => void act(() => api.decide(c.id, 'skipped'))}>Skip</button>
              {c.synthetic && <span className="badge">Synthetic</span>}
            </div>
          )}
        </div>
      ))}
      {unmatched > 0 && <p className="muted">{unmatched} feature{unmatched === 1 ? ' has' : 's have'} no OpenStreetMap kerb node within 8 m and would need mapping by hand.</p>}
      <div className="card">
        <p>
          {approved.length} approved edit{approved.length === 1 ? '' : 's'}
          {syntheticApproved > 0 ? `, ${syntheticApproved} resting on synthetic demo reports. Those are left out unless you tick the box, and then the file is marked DO NOT UPLOAD.` : '.'}
        </p>
        <label className="row toggle">
          <input type="checkbox" checked={allowSynthetic} onChange={(e) => setAllowSynthetic(e.target.checked)} data-testid="allow-synthetic" />
          <span>Include synthetic demo edits, to see the file format</span>
        </label>
        <div className="inline">
          <button className="pill" onClick={() => void downloadDraft()} data-testid="download-osc">Download OsmChange draft</button>
          <button className="pill ghost" onClick={async () => download('curbcut-features.geojson', await api.dataset(), 'application/geo+json')} data-testid="download-dataset">Download dataset (GeoJSON, ODbL)</button>
        </div>
        {note && <p className="muted" data-testid="draft-note">{note}</p>}
      </div>
    </div>
  )
}
