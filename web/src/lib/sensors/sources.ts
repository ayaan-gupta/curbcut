import type { Fix } from './recorder'
import type { MotionSample } from './resample'

export interface SensorSource {
  kind: 'device' | 'simulated'
  /** Resolves once samples are flowing. Rejects with a message fit to show the user. */
  start(onMotion: (s: MotionSample) => void, onFix: (f: Fix) => void, onEnd: () => void): Promise<void>
  stop(): void
}

type MotionPermission = { requestPermission?: () => Promise<'granted' | 'denied'> }

export function deviceSensorsAvailable(): boolean {
  return typeof window !== 'undefined' && 'DeviceMotionEvent' in window && 'geolocation' in navigator && ('ontouchstart' in window || navigator.maxTouchPoints > 0)
}

/**
 * The phone's own accelerometer, gyroscope and GPS, through the browser.
 * Both need the person's permission and both are only read between start()
 * and stop(). A web page cannot keep reading them once it is closed or the
 * screen locks, and this app stops recording when the page is hidden, so
 * there is no background tracking to switch off.
 */
export function deviceSource(): SensorSource {
  let watch: number | null = null
  let handler: ((e: DeviceMotionEvent) => void) | null = null
  let wakeLock: { release: () => Promise<void> } | null = null
  let onHide: (() => void) | null = null
  return {
    kind: 'device',
    async start(onMotion, onFix, onEnd) {
      const motion = DeviceMotionEvent as unknown as MotionPermission
      if (typeof motion.requestPermission === 'function') {
        const answer = await motion.requestPermission().catch(() => 'denied' as const)
        if (answer !== 'granted') throw new Error('Motion sensor access was not allowed. Recording needs it to feel the ground.')
      }
      await new Promise<void>((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(() => resolve(), (e) => reject(new Error(e.code === e.PERMISSION_DENIED ? 'Location access was not allowed. Recording needs it to place what it feels.' : 'No location fix yet. Try again outdoors.')), { enableHighAccuracy: true, timeout: 15000 })
      })
      const deg = Math.PI / 180
      handler = (e) => {
        const a = e.accelerationIncludingGravity
        if (!a || a.x == null || a.y == null || a.z == null) return
        const r = e.rotationRate
        onMotion({ t: e.timeStamp, ax: a.x, ay: a.y, az: a.z, gx: (r?.beta ?? 0) * deg, gy: (r?.gamma ?? 0) * deg, gz: (r?.alpha ?? 0) * deg })
      }
      window.addEventListener('devicemotion', handler)
      watch = navigator.geolocation.watchPosition(
        (p) => onFix({ t: performance.now(), lon: p.coords.longitude, lat: p.coords.latitude, accuracy: p.coords.accuracy }),
        () => undefined,
        { enableHighAccuracy: true, maximumAge: 1000 },
      )
      onHide = () => { if (document.hidden) onEnd() }
      document.addEventListener('visibilitychange', onHide)
      try {
        wakeLock = await (navigator as unknown as { wakeLock?: { request: (t: string) => Promise<{ release: () => Promise<void> }> } }).wakeLock?.request('screen') ?? null
      } catch { wakeLock = null }
    },
    stop() {
      if (handler) window.removeEventListener('devicemotion', handler)
      if (watch !== null) navigator.geolocation.clearWatch(watch)
      if (onHide) document.removeEventListener('visibilitychange', onHide)
      void wakeLock?.release()
      handler = null
      watch = null
    },
  }
}

interface DemoWalk {
  rate_hz: number
  duration_s: number
  fixes: Fix[]
  samples: number[]
}

/**
 * Replays a simulated walk (public/demo/walk.json, SYNTHETIC) through the
 * same pipeline as real sensors, faster than real time. Used on machines
 * with no motion sensors, and labelled as simulated wherever it shows.
 */
export function simulatedSource(speed = 8): SensorSource {
  let timer: number | null = null
  return {
    kind: 'simulated',
    async start(onMotion, onFix, onEnd) {
      const response = await fetch('/demo/walk.json')
      if (!response.ok) throw new Error('The simulated walk could not be loaded.')
      const walk = (await response.json()) as DemoWalk
      const stepMs = 1000 / walk.rate_hz
      const total = walk.samples.length / 6
      let i = 0
      let nextFix = 0
      const began = performance.now()
      timer = window.setInterval(() => {
        const upTo = Math.min(total, Math.floor(((performance.now() - began) * speed) / stepMs))
        for (; i < upTo; i++) {
          const t = i * stepMs
          while (nextFix < walk.fixes.length && walk.fixes[nextFix]!.t <= t) onFix(walk.fixes[nextFix++]!)
          const o = i * 6
          onMotion({ t, ax: walk.samples[o]!, ay: walk.samples[o + 1]!, az: walk.samples[o + 2]!, gx: walk.samples[o + 3]!, gy: walk.samples[o + 4]!, gz: walk.samples[o + 5]! })
        }
        if (i >= total) {
          if (timer !== null) window.clearInterval(timer)
          timer = null
          onEnd()
        }
      }, 50)
    },
    stop() {
      if (timer !== null) window.clearInterval(timer)
      timer = null
    },
  }
}
