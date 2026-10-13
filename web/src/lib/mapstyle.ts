import type { StyleSpecification } from 'maplibre-gl'
import { api } from './api'

export const REMOTE_STYLE = (import.meta.env.VITE_MAP_STYLE_URL as string | undefined) ?? 'https://tiles.openfreemap.org/styles/positron'

export function wantsOffline(): boolean {
  return REMOTE_STYLE === 'offline' || new URLSearchParams(location.search).get('map') === 'offline'
}

/**
 * A basemap that needs no tile server: the pilot area's roads, buildings and
 * parks from the committed OpenStreetMap extract, served by our own API.
 * Used when VITE_MAP_STYLE_URL is "offline", when ?map=offline is in the
 * address, and as the fallback when the remote style cannot be reached.
 */
export async function offlineStyle(): Promise<StyleSpecification> {
  const base = await api.basemap()
  const road = (cls: string[], width: number, color = '#ffffff') => ({
    id: `road-${cls[0]}`,
    type: 'line' as const,
    source: 'roads',
    filter: ['in', ['get', 'cls'], ['literal', cls]] as never,
    layout: { 'line-cap': 'round' as const, 'line-join': 'round' as const },
    paint: { 'line-color': color, 'line-width': ['interpolate', ['exponential', 1.6], ['zoom'], 14, width * 0.35, 18, width * 2.2] as never },
  })
  return {
    version: 8,
    sources: {
      roads: { type: 'geojson', data: base.roads },
      areas: { type: 'geojson', data: base.areas },
    },
    layers: [
      { id: 'background', type: 'background', paint: { 'background-color': '#efede7' } },
      { id: 'parks', type: 'fill', source: 'areas', filter: ['in', ['get', 'cls'], ['literal', ['park', 'pitch', 'playground']]], paint: { 'fill-color': '#d7e4cc' } },
      { id: 'water', type: 'fill', source: 'areas', filter: ['==', ['get', 'cls'], 'water'], paint: { 'fill-color': '#bcd6e6' } },
      { id: 'buildings', type: 'fill', source: 'areas', filter: ['==', ['get', 'cls'], 'building'], paint: { 'fill-color': '#e2ded5', 'fill-outline-color': '#d3cec3' } },
      road(['service', 'living_street'], 3.5, '#f8f7f3'),
      road(['residential', 'unclassified'], 6),
      road(['tertiary', 'secondary', 'primary'], 9),
    ],
  }
}
