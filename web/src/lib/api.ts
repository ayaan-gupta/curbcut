import type { FeatureCollection, LineString, Point } from 'geojson'
import { deviceId } from './device'
import type { LonLat } from './geo'

export type Profile = 'manual_wheelchair' | 'powered_wheelchair' | 'stroller' | 'walking'
export type Category = 'curb' | 'surface' | 'slope'

export interface FeatureProps {
  id: number
  category: Category
  state: string
  kind: string
  confidence: number
  agreement: number
  reporters: number
  reports: number
  published: boolean
  reviewed: 'approved' | 'rejected' | null
  value: number | null
  material: string | null
  synthetic: boolean
  updated_at: string
}
export type Features = FeatureCollection<Point, FeatureProps>

export interface Meta {
  pilot: { name: string; bbox: { south: number; west: number; north: number; east: number }; attribution: string; license: string }
  publish_threshold: number
  cluster_radius_m: number
  review_needs_token: boolean
  profiles: Record<Profile, { label: string; max_incline: number | null; max_kerb_cm: number | null }>
  demo: { origin: LonLat; destination: LonLat; missing_curb: LonLat } | null
  counts: { reports: number; features: number; published: number; in_review: number }
}

export interface ReportIn {
  category: Category
  state: string
  lon: number
  lat: number
  accuracy_m: number
  source: 'photo' | 'sensor' | 'manual'
  model_confidence: number
  value?: number | null
  material?: string | null
  recorded_at?: string
  simulated?: boolean
  faces_blurred?: number
  regions_blurred?: number
}

export interface PublicReport {
  id: number
  category: Category
  state: string
  lon: number
  lat: number
  source: string
  model_confidence: number
  value: number | null
  created_at: string
  status: string
  flags: string[]
  has_photo: boolean
  synthetic: boolean
}

export interface SubmitResult {
  report: { id: number; status: string; flags: string[] }
  feature: { id: number; confidence: number; published: boolean; reporters: number; state: string } | null
  merged: boolean
  message: string
}

export interface Barrier { edge_id: string; reason: string; lon: number; lat: number }
export interface RouteLeg { distance_m: number; geometry: LineString; barriers: Barrier[]; max_incline: number | null; cautions: string[] }
export interface RouteResponse {
  status: 'ok' | 'no_accessible_route' | 'unreachable'
  profile: Profile
  route: RouteLeg | null
  shortest: RouteLeg | null
  detour_m: number | null
  message: string
}

export type QueueFeature = FeatureProps & { lon: number; lat: number }
export interface ReviewQueue { held_reports: PublicReport[]; conflicts: QueueFeature[]; below_threshold: QueueFeature[] }

export interface Candidate {
  id: string
  feature_id: number
  element: 'node' | 'way'
  osm_id: number
  key: string
  value: string
  current: string | null
  kind: 'add' | 'change' | 'confirm'
  distance_m: number
  confidence: number
  reporters: number
  synthetic: boolean
  needs_direction: boolean
  decision: 'pending' | 'approved' | 'skipped'
  note: string
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
  }
}

let reviewerToken = ''
export function setReviewerToken(token: string): void {
  reviewerToken = token
}

async function call<T>(path: string, init: RequestInit = {}, as: 'json' | 'text' = 'json'): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set('X-Device-Id', deviceId())
  if (reviewerToken) headers.set('Authorization', `Bearer ${reviewerToken}`)
  const response = await fetch(`/api${path}`, { ...init, headers })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown }
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* keep the status text */ }
    throw new ApiError(detail, response.status)
  }
  return (as === 'json' ? response.json() : response.text()) as Promise<T>
}

const json = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  meta: () => call<Meta>('/meta'),
  basemap: () => call<{ roads: FeatureCollection; areas: FeatureCollection; attribution: string }>('/basemap'),
  features: (include: 'published' | 'all') => call<Features>(`/features?include=${include}`),
  network: (profile: Profile) => call<FeatureCollection<LineString, { code: string; passable: boolean; reason: string }>>(`/network?profile=${profile}`),
  route: (origin: LonLat, destination: LonLat, profile: Profile) => call<RouteResponse>('/route', json({ origin, destination, profile })),
  submit(report: ReportIn, photo?: Blob): Promise<SubmitResult> {
    const form = new FormData()
    form.set('report', JSON.stringify(report))
    if (photo) form.set('photo', photo, 'curb.jpg')
    return call<SubmitResult>('/reports', { method: 'POST', body: form })
  },
  submitBatch: (reports: ReportIn[]) => call<{ accepted: number; rate_limited: string | null; results: SubmitResult[] }>('/reports/batch', json(reports)),
  review: () => call<ReviewQueue>('/review'),
  reviewReport: (id: number, action: 'accept' | 'reject') => call<PublicReport>(`/review/reports/${id}`, json({ action })),
  reviewFeature: (id: number, action: 'approve' | 'reject' | 'clear') => call<unknown>(`/review/features/${id}`, json({ action })),
  candidates: () => call<{ candidates: Candidate[]; unmatched: Array<{ feature_id: number; wants: string; why: string }>; export_threshold: number }>('/export/candidates'),
  decide: (id: string, decision: Candidate['decision'], incline_sign?: 1 | -1) => call<Candidate>(`/export/candidates/${encodeURIComponent(id)}`, json({ decision, incline_sign: incline_sign ?? null })),
  osmChange: (allowSynthetic: boolean) => call<string>(`/export/osmchange?allow_synthetic=${allowSynthetic}`, {}, 'text'),
  dataset: () => call<string>('/export/dataset.geojson', {}, 'text'),
}

export function download(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}
