// Curbcut service worker: app shell and on-device models from cache, the API always from the network.
const CACHE = 'curbcut-v1'
const PRECACHE = ['/', '/manifest.webmanifest', '/models/roughness.onnx', '/models/curb-photo.onnx', '/models/face-ultraface-320.onnx', '/ort/ort-wasm-simd-threaded.wasm', '/ort/ort-wasm-simd-threaded.mjs']

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting()))
})

self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()))
})

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url)
  if (event.request.method !== 'GET' || url.origin !== location.origin || url.pathname.startsWith('/api/')) return
  event.respondWith(
    caches.match(event.request).then((hit) => {
      const fresh = fetch(event.request)
        .then((response) => {
          if (response.ok) caches.open(CACHE).then((cache) => cache.put(event.request, response.clone()))
          return response.clone()
        })
        .catch(() => hit)
      return hit || fresh
    }),
  )
})
