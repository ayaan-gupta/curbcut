const R = 6371008.8

export type LonLat = [number, number]

export function haversine(a: LonLat, b: LonLat): number {
  const toRad = Math.PI / 180
  const p1 = a[1] * toRad
  const p2 = b[1] * toRad
  const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(((b[0] - a[0]) * toRad) / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

export function pathLength(points: LonLat[]): number {
  let total = 0
  for (let i = 1; i < points.length; i++) total += haversine(points[i - 1]!, points[i]!)
  return total
}

/**
 * Distance actually covered by a GPS trace. Fixes wander a couple of metres
 * even when standing still, and summing every wiggle triples the length of a
 * slow roll, so a fix only counts once it is at least `minStep` metres from
 * the last one that counted.
 */
export function travelled(points: LonLat[], minStep = 8): number {
  let total = 0
  let anchor = points[0]
  for (const p of points) {
    const d = anchor ? haversine(anchor, p) : 0
    if (d >= minStep) {
      total += d
      anchor = p
    }
  }
  return total
}

export function formatDistance(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(metres >= 10_000 ? 0 : 1)} km` : `${Math.round(metres)} m`
}

/** Minutes at a pace: about 1.1 m/s for a manual chair or a stroller, 1.3 m/s on foot or in a powered chair. */
export function formatMinutes(metres: number, metresPerSecond: number): string {
  const minutes = Math.max(1, Math.round(metres / metresPerSecond / 60))
  return `${minutes} min`
}
