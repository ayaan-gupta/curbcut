import type { FeatureCollection } from 'geojson'
import { useCallback, useEffect, useState } from 'react'
import { api, type Features, type Meta, type Profile, type RouteResponse } from './lib/api'
import type { LonLat } from './lib/geo'
import { MapTab } from './components/MapTab'
import { MapView, type TaggedDot } from './components/MapView'
import { Onboarding } from './components/Onboarding'
import { PhotoTab } from './components/PhotoTab'
import { RecordTab } from './components/RecordTab'
import { ReviewTab } from './components/ReviewTab'

type Tab = 'map' | 'record' | 'photo' | 'review'
const TABS: Array<{ key: Tab; label: string; icon: string }> = [
  { key: 'map', label: 'Map', icon: 'M9 4 3 6.5v13.5l6-2.5 6 2.5 6-2.5V4l-6 2.5zM9 4v13.5M15 6.5V20' },
  { key: 'record', label: 'Record', icon: 'M3 13h3l2.5-6 4 11 3-8 1.5 3H21' },
  { key: 'photo', label: 'Photo', icon: 'M4 8h3.5l1.5-2.5h6L16.5 8H20v11H4zM12 16.5a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4z' },
  { key: 'review', label: 'Review', icon: 'M5 4h14v16H5zM8.5 12.5l2.5 2.5 4.5-5' },
]
const params = new URLSearchParams(location.search)

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>((params.get('tab') as Tab | null) ?? 'map')
  const [welcome, setWelcome] = useState(() => params.get('onboarding') !== '0' && localStorage.getItem('curbcut.welcomed') !== '1')
  const [features, setFeatures] = useState<Features | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [profile, setProfile] = useState<Profile>('manual_wheelchair')
  const [showClosed, setShowClosed] = useState(false)
  const [network, setNetwork] = useState<FeatureCollection | null>(null)
  const [origin, setOrigin] = useState<LonLat | null>(null)
  const [destination, setDestination] = useState<LonLat | null>(null)
  const [placing, setPlacing] = useState<'origin' | 'destination' | 'photo' | null>(null)
  const [route, setRoute] = useState<RouteResponse | null>(null)
  const [routeError, setRouteError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [trace, setTrace] = useState<LonLat[]>([])
  const [tagged, setTagged] = useState<TaggedDot[]>([])
  const [photoPoint, setPhotoPoint] = useState<LonLat | null>(null)
  const [version, setVersion] = useState(0)
  const bump = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    api.meta().then(setMeta).catch((e: unknown) => setFailed(e instanceof Error ? e.message : String(e)))
  }, [])
  useEffect(() => {
    api.features(showAll || tab === 'review' ? 'all' : 'published').then(setFeatures).catch(() => undefined)
  }, [showAll, tab, version])
  useEffect(() => {
    if (!showClosed) return setNetwork(null)
    api.network(profile).then(setNetwork).catch(() => undefined)
  }, [showClosed, profile, version])
  useEffect(() => {
    if (!origin || !destination) return setRoute(null)
    let stale = false
    setRouteError(null)
    api.route(origin, destination, profile).then((r) => !stale && setRoute(r)).catch((e: unknown) => !stale && setRouteError(e instanceof Error ? e.message : String(e)))
    return () => { stale = true }
  }, [origin, destination, profile, version])

  const onMapClick = (at: LonLat) => {
    if (placing === 'origin') { setOrigin(at); setPlacing(destination ? null : 'destination') }
    else if (placing === 'destination') { setDestination(at); setPlacing(null) }
    else if (placing === 'photo' || tab === 'photo') { setPhotoPoint(at); setPlacing(null) }
    else setSelected(null)
  }

  if (failed) return <div className="fatal">Curbcut could not reach its API ({failed}). Start it with <code>make up</code> or <code>make dev</code>.</div>
  if (!meta) return <div className="fatal">Loading…</div>

  return (
    <div className={`app tab-${tab}`}>
      <main className="mapwrap">
        <MapView
          meta={meta}
          features={features}
          network={network}
          route={tab === 'map' ? route : null}
          origin={tab === 'map' ? origin : null}
          destination={tab === 'map' ? destination : null}
          trace={tab === 'record' ? trace : []}
          tagged={tab === 'record' ? tagged : []}
          picked={tab === 'photo' ? photoPoint : null}
          onClick={onMapClick}
          onFeature={(id) => { setSelected(id); setTab('map') }}
        />
        {placing && <div className="toast">Tap the map to place {placing === 'photo' ? 'the photo' : placing === 'origin' ? 'the start' : 'the end'}</div>}
      </main>
      <section className="panel" aria-label={tab}>
        <header>
          <h1>Curbcut</h1>
          <span className="muted">{meta.pilot.name}</span>
        </header>
        <div className="content" data-testid={`tab-${tab}`}>
          {tab === 'map' && (
            <MapTab
              meta={meta}
              profile={profile}
              setProfile={setProfile}
              origin={origin}
              destination={destination}
              placing={placing === 'photo' ? null : placing}
              setPlacing={setPlacing}
              route={route}
              routeError={routeError}
              showClosed={showClosed}
              setShowClosed={setShowClosed}
              showAll={showAll}
              setShowAll={setShowAll}
              features={features}
              selected={selected}
              clearSelected={() => setSelected(null)}
              useDemo={() => { if (meta.demo) { setOrigin(meta.demo.origin); setDestination(meta.demo.destination); setPlacing(null) } }}
              clear={() => { setOrigin(null); setDestination(null); setRoute(null); setPlacing(null) }}
            />
          )}
          {tab === 'record' && <RecordTab features={features} onTrace={(t, d) => { setTrace(t); setTagged(d) }} onUploaded={bump} />}
          {tab === 'photo' && <PhotoTab picked={photoPoint} onNeedPoint={() => setPlacing('photo')} onSubmitted={bump} />}
          {tab === 'review' && <ReviewTab meta={meta} onChanged={bump} />}
        </div>
      </section>
      <nav className="tabs" aria-label="Sections">
        {TABS.map((t) => (
          <button key={t.key} className={tab === t.key ? 'on' : ''} onClick={() => { setTab(t.key); setPlacing(null) }} data-testid={`nav-${t.key}`} aria-current={tab === t.key}>
            <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d={t.icon} /></svg>
            <span>{t.label}</span>
          </button>
        ))}
      </nav>
      {welcome && <Onboarding onDone={() => { localStorage.setItem('curbcut.welcomed', '1'); setWelcome(false) }} />}
    </div>
  )
}
