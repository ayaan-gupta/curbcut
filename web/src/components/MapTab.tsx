import type { Features, Meta, Profile, RouteResponse } from '../lib/api'
import { formatDistance, formatMinutes, type LonLat } from '../lib/geo'
import { iconUrl } from './MapView'

export const PROFILE_LABEL: Record<Profile, string> = { manual_wheelchair: 'Manual chair', powered_wheelchair: 'Powered chair', stroller: 'Stroller', walking: 'On foot' }
const PACE: Record<Profile, number> = { manual_wheelchair: 1.0, powered_wheelchair: 1.3, stroller: 1.1, walking: 1.3 }
export const KIND_LABEL: Record<string, string> = {
  curb_present: 'Curb cut',
  curb_missing: 'No curb cut',
  curb_damaged: 'Damaged ramp',
  curb_blocked: 'Blocked ramp',
  surface_rough: 'Rough surface',
  surface_bump: 'Bump',
  slope_steep: 'Steep slope',
}

interface Props {
  meta: Meta
  profile: Profile
  setProfile: (p: Profile) => void
  origin: LonLat | null
  destination: LonLat | null
  placing: 'origin' | 'destination' | null
  setPlacing: (p: 'origin' | 'destination' | null) => void
  route: RouteResponse | null
  routeError: string | null
  showClosed: boolean
  setShowClosed: (v: boolean) => void
  showAll: boolean
  setShowAll: (v: boolean) => void
  features: Features | null
  selected: number | null
  clearSelected: () => void
  useDemo: () => void
  clear: () => void
}

export function MapTab(p: Props) {
  const limits = p.meta.profiles[p.profile]
  const feature = p.selected != null ? p.features?.features.find((f) => f.properties.id === p.selected)?.properties : undefined
  return (
    <div className="stack">
      <div className="segmented" role="tablist" aria-label="Who is travelling">
        {(['manual_wheelchair', 'powered_wheelchair', 'stroller'] as Profile[]).map((key) => (
          <button key={key} role="tab" aria-selected={p.profile === key} className={p.profile === key ? 'on' : ''} onClick={() => p.setProfile(key)} data-testid={`profile-${key}`}>
            {PROFILE_LABEL[key]}
          </button>
        ))}
      </div>
      <p className="hint">
        Never routes over a slope above {limits.max_incline}% or a kerb above {limits.max_kerb_cm} cm{p.profile === 'stroller' ? '' : ', steps or loose surfaces'}.
      </p>

      <div className="rows">
        <button className={`row ${p.placing === 'origin' ? 'active' : ''}`} onClick={() => p.setPlacing(p.placing === 'origin' ? null : 'origin')} data-testid="set-origin">
          <span className="dot dot-a">A</span>
          <span>{p.placing === 'origin' ? 'Tap the map for the start' : p.origin ? 'Start set' : 'Choose a start'}</span>
        </button>
        <button className={`row ${p.placing === 'destination' ? 'active' : ''}`} onClick={() => p.setPlacing(p.placing === 'destination' ? null : 'destination')} data-testid="set-destination">
          <span className="dot dot-b">B</span>
          <span>{p.placing === 'destination' ? 'Tap the map for the end' : p.destination ? 'End set' : 'Choose an end'}</span>
        </button>
      </div>
      <div className="inline">
        {p.meta.demo && (
          <button className="pill" onClick={p.useDemo} data-testid="demo-route">
            Try the demo walk
          </button>
        )}
        {(p.origin || p.destination) && (
          <button className="pill ghost" onClick={p.clear}>
            Clear
          </button>
        )}
      </div>

      {p.routeError && <div className="card warn">{p.routeError}</div>}
      {p.route && (
        <div className={`card ${p.route.status === 'ok' ? '' : 'warn'}`} data-testid="route-result" data-status={p.route.status}>
          {p.route.status === 'ok' && p.route.route ? (
            <>
              <div className="big">
                {formatDistance(p.route.route.distance_m)} <span className="muted">about {formatMinutes(p.route.route.distance_m, PACE[p.profile])}</span>
              </div>
              {p.route.shortest && p.route.shortest.barriers.length > 0 ? (
                <p>
                  {formatDistance(Math.max(0, p.route.detour_m ?? 0))} longer than the shortest walk (dashed), which crosses:
                </p>
              ) : (
                <p>The shortest walk is clear for this profile.</p>
              )}
            </>
          ) : (
            <>
              <div className="big">No accessible route</div>
              <p>{p.route.status === 'unreachable' ? p.route.message : 'Every way between these points crosses a barrier for this profile. The dashed line is the shortest walk, with what blocks it:'}</p>
            </>
          )}
          <ul className="reasons">
            {[...new Set((p.route.shortest?.barriers ?? []).map((b) => b.reason))].map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
          {p.route.route?.cautions.map((c) => (
            <p key={c} className="muted">
              Heads up: {c}.
            </p>
          ))}
          {p.route.route?.max_incline != null && <p className="muted">Steepest stretch on this route: {p.route.route.max_incline.toFixed(0)}%.</p>}
        </div>
      )}

      {feature && (
        <div className="card" data-testid="feature-card">
          <div className="feature-head">
            {iconUrl(feature.kind) && <img src={iconUrl(feature.kind)} width={32} height={32} alt="" />}
            <div>
              <div className="big small">{KIND_LABEL[feature.kind] ?? feature.kind}</div>
              <div className="muted">
                {feature.reporters} reporter{feature.reporters === 1 ? '' : 's'}, {feature.published ? 'on the open map' : 'not yet published'}
              </div>
            </div>
            <button className="close" onClick={p.clearSelected} aria-label="Close">
              ×
            </button>
          </div>
          <div className="meter" aria-label={`Confidence ${feature.confidence.toFixed(2)}`}>
            <div style={{ width: `${Math.round(feature.confidence * 100)}%` }} />
            <i style={{ left: `${p.meta.publish_threshold * 100}%` }} />
          </div>
          <p className="muted">
            Confidence {feature.confidence.toFixed(2)} of 1. Published at {p.meta.publish_threshold.toFixed(2)}. Agreement {Math.round(feature.agreement * 100)}%.
            {feature.value != null && feature.category === 'slope' ? ` Measured ${feature.value.toFixed(0)}%.` : ''}
          </p>
          {feature.synthetic && <span className="badge">Synthetic demo data</span>}
        </div>
      )}

      <label className="row toggle">
        <input type="checkbox" checked={p.showClosed} onChange={(e) => p.setShowClosed(e.target.checked)} data-testid="toggle-network" />
        <span>Show what is closed to a {PROFILE_LABEL[p.profile].toLowerCase()}</span>
      </label>
      <label className="row toggle">
        <input type="checkbox" checked={p.showAll} onChange={(e) => p.setShowAll(e.target.checked)} />
        <span>Show reports still below the publishing threshold</span>
      </label>

      <div className="legend">
        {Object.entries(KIND_LABEL).map(([kind, label]) => (
          <span key={kind}>
            <img src={iconUrl(kind)} width={20} height={20} alt="" /> {label}
          </span>
        ))}
      </div>
      <p className="fine">
        Pilot area: {p.meta.pilot.name}. Footways, kerbs and surfaces are real OpenStreetMap data. {p.meta.demo ? 'The reports on it are synthetic demo data.' : ''}
      </p>
    </div>
  )
}
