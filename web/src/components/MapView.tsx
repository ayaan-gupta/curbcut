import type { FeatureCollection } from 'geojson'
import maplibregl, { type GeoJSONSource, type Map as MlMap } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { useEffect, useRef } from 'react'
import art from 'virtual:art'
import type { Features, Meta, RouteResponse } from '../lib/api'
import type { LonLat } from '../lib/geo'
import { offlineStyle, REMOTE_STYLE, wantsOffline } from '../lib/mapstyle'

export const KIND_ICON: Record<string, string> = {
  curb_present: 'icons/curb-present',
  curb_missing: 'icons/curb-missing',
  curb_damaged: 'icons/curb-damaged',
  curb_blocked: 'icons/curb-blocked',
  surface_rough: 'icons/surface-rough',
  surface_bump: 'icons/surface-bump',
  slope_steep: 'icons/slope-steep',
}

export function iconUrl(kind: string): string | undefined {
  const name = KIND_ICON[kind]
  return name ? art[name]?.url : undefined
}

export interface TaggedDot {
  lon: number
  lat: number
  state: string
}

interface Props {
  meta: Meta
  features: Features | null
  network: FeatureCollection | null
  route: RouteResponse | null
  origin: LonLat | null
  destination: LonLat | null
  trace: LonLat[]
  tagged: TaggedDot[]
  picked: LonLat | null
  onClick: (at: LonLat) => void
  onFeature: (id: number) => void
}

const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] }
const line = (coords: LonLat[]): FeatureCollection => ({ type: 'FeatureCollection', features: coords.length > 1 ? [{ type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: coords } }] : [] })

async function addIcons(map: MlMap): Promise<void> {
  await Promise.all(
    Object.entries(KIND_ICON).map(async ([kind, name]) => {
      const url = art[name]?.url
      if (!url || map.hasImage(kind)) return
      const img = new Image(48, 48)
      img.src = url
      await img.decode()
      if (!map.hasImage(kind)) map.addImage(kind, img, { pixelRatio: 2 })
    }),
  )
}

function addLayers(map: MlMap): void {
  for (const id of ['network', 'shortest', 'route', 'trace', 'tagged', 'features', 'barriers']) map.addSource(id, { type: 'geojson', data: EMPTY })
  const round = { 'line-cap': 'round' as const, 'line-join': 'round' as const }
  map.addLayer({ id: 'network-ok', type: 'line', source: 'network', filter: ['==', ['get', 'passable'], true], layout: round, paint: { 'line-color': ['case', ['==', ['get', 'code'], 'caution'], '#b25a00', '#5f7a68'], 'line-width': 2, 'line-opacity': 0.55 } })
  map.addLayer({ id: 'network-closed', type: 'line', source: 'network', filter: ['==', ['get', 'passable'], false], layout: round, paint: { 'line-color': '#c4281c', 'line-width': 3, 'line-dasharray': [0.6, 1.6] } })
  map.addLayer({ id: 'shortest', type: 'line', source: 'shortest', layout: round, paint: { 'line-color': '#6b7280', 'line-width': 3, 'line-dasharray': [1, 2] } })
  map.addLayer({ id: 'route-casing', type: 'line', source: 'route', layout: round, paint: { 'line-color': '#111111', 'line-width': 9 } })
  map.addLayer({ id: 'route', type: 'line', source: 'route', layout: round, paint: { 'line-color': '#ffd60a', 'line-width': 5 } })
  map.addLayer({ id: 'trace', type: 'line', source: 'trace', layout: round, paint: { 'line-color': '#1d4e89', 'line-width': 4, 'line-opacity': 0.8 } })
  map.addLayer({ id: 'tagged', type: 'circle', source: 'tagged', paint: { 'circle-radius': 7, 'circle-color': ['match', ['get', 'state'], 'bump', '#c4281c', 'rough', '#b25a00', '#0a7b45'], 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 } })
  map.addLayer({ id: 'barriers', type: 'circle', source: 'barriers', paint: { 'circle-radius': 13, 'circle-color': 'rgba(196,40,28,0.16)', 'circle-stroke-color': '#c4281c', 'circle-stroke-width': 2.5 } })
  map.addLayer({
    id: 'features',
    type: 'symbol',
    source: 'features',
    layout: { 'icon-image': ['get', 'kind'], 'icon-size': ['interpolate', ['linear'], ['zoom'], 14, 0.8, 17, 1.25], 'icon-allow-overlap': true },
    paint: { 'icon-opacity': ['case', ['get', 'published'], 1, 0.4] },
  })
}

/** The map. One MapLibre instance for the life of the app; props are pushed into GeoJSON sources. */
export function MapView(props: Props) {
  const holder = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MlMap | null>(null)
  const ready = useRef(false)
  const latest = useRef(props)
  latest.current = props
  const markers = useRef<{ a?: maplibregl.Marker; b?: maplibregl.Marker; p?: maplibregl.Marker }>({})

  const sync = () => {
    const map = mapRef.current
    if (!map || !ready.current) return
    const p = latest.current
    const set = (id: string, data: FeatureCollection) => (map.getSource(id) as GeoJSONSource | undefined)?.setData(data)
    set('features', (p.features as FeatureCollection | null) ?? EMPTY)
    set('network', p.network ?? EMPTY)
    set('route', p.route?.route ? { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: p.route.route.geometry }] } : EMPTY)
    const showShortest = p.route?.shortest && (p.route.status !== 'ok' || (p.route.detour_m ?? 0) > 5)
    set('shortest', showShortest ? { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: p.route!.shortest!.geometry }] } : EMPTY)
    set('barriers', { type: 'FeatureCollection', features: (p.route?.shortest?.barriers ?? []).map((b) => ({ type: 'Feature', properties: { reason: b.reason }, geometry: { type: 'Point', coordinates: [b.lon, b.lat] } })) })
    set('trace', line(p.trace))
    set('tagged', { type: 'FeatureCollection', features: p.tagged.map((t) => ({ type: 'Feature', properties: { state: t.state }, geometry: { type: 'Point', coordinates: [t.lon, t.lat] } })) })
    const place = (key: 'a' | 'b' | 'p', at: LonLat | null, label: string, cls: string) => {
      if (!at) {
        markers.current[key]?.remove()
        markers.current[key] = undefined
        return
      }
      if (!markers.current[key]) {
        const el = document.createElement('div')
        el.className = `pin ${cls}`
        const text = document.createElement('span')
        text.textContent = label
        el.appendChild(text)
        markers.current[key] = new maplibregl.Marker({ element: el, anchor: 'bottom' }).setLngLat(at).addTo(map)
      } else markers.current[key]!.setLngLat(at)
    }
    place('a', p.origin, 'A', 'pin-a')
    place('b', p.destination, 'B', 'pin-b')
    place('p', p.picked, '', 'pin-pick')
  }

  useEffect(() => {
    let cancelled = false
    const b = props.meta.pilot.bbox
    const start = async () => {
      const offline = wantsOffline()
      const style = offline ? await offlineStyle() : REMOTE_STYLE
      if (cancelled || !holder.current) return
      const map = new maplibregl.Map({
        container: holder.current,
        style,
        bounds: [[b.west, b.south], [b.east, b.north]],
        fitBoundsOptions: { padding: 12 },
        maxBounds: [[b.west - 0.02, b.south - 0.012], [b.east + 0.02, b.north + 0.012]],
        attributionControl: { compact: true, customAttribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' },
      })
      mapRef.current = map
      ;(window as unknown as { __map?: MlMap }).__map = map
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
      let fellBack = offline
      const fallBack = async () => {
        if (fellBack || ready.current) return
        fellBack = true
        map.setStyle(await offlineStyle())
      }
      // If the tile host cannot be reached, draw the pilot area from our own data instead of a blank map.
      const timer = window.setTimeout(() => void fallBack(), 7000)
      map.on('error', (e) => { if (!ready.current && /style|fetch|load/i.test(String(e.error?.message ?? ''))) void fallBack() })
      map.on('style.load', async () => {
        await addIcons(map)
        if (cancelled || map.getSource('features')) return
        addLayers(map)
        ready.current = true
        window.clearTimeout(timer)
        holder.current?.setAttribute('data-ready', 'true')
        sync()
      })
      map.on('click', (e) => {
        const hit = map.getLayer('features') ? map.queryRenderedFeatures(e.point, { layers: ['features'] })[0] : undefined
        if (hit?.properties?.id != null) latest.current.onFeature(Number(hit.properties.id))
        else latest.current.onClick([e.lngLat.lng, e.lngLat.lat])
      })
      map.on('mouseenter', 'features', () => (map.getCanvas().style.cursor = 'pointer'))
      map.on('mouseleave', 'features', () => (map.getCanvas().style.cursor = ''))
    }
    void start()
    return () => {
      cancelled = true
      ready.current = false
      mapRef.current?.remove()
      mapRef.current = null
      markers.current = {}
    }
    // The map is created once. Later prop changes go through sync().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(sync)

  useEffect(() => {
    const map = mapRef.current
    const geometry = props.route?.route?.geometry ?? props.route?.shortest?.geometry
    if (!map || !geometry) return
    const bounds = new maplibregl.LngLatBounds()
    for (const c of geometry.coordinates) bounds.extend(c as LonLat)
    map.fitBounds(bounds, { padding: 70, maxZoom: 17.5, duration: 600 })
  }, [props.route])

  useEffect(() => {
    const map = mapRef.current
    const last = props.trace[props.trace.length - 1]
    if (map && last) map.easeTo({ center: last, zoom: Math.max(map.getZoom(), 17.2), duration: 300 })
  }, [props.trace])

  return <div ref={holder} className="map" data-testid="map" />
}
