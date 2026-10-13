import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const assets = path.resolve(import.meta.dirname, '..', 'assets')
const ortDist = path.resolve(import.meta.dirname, 'node_modules', 'onnxruntime-web', 'dist')
const ORT_FILES = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']
const MIME: Record<string, string> = { '.svg': 'image/svg+xml', '.png': 'image/png', '.webp': 'image/webp', '.jpg': 'image/jpeg', '.wasm': 'application/wasm', '.mjs': 'text/javascript' }

function walk(dir: string, base = dir): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name)
    return statSync(full).isDirectory() ? walk(full, base) : [path.relative(base, full)]
  })
}

/**
 * Art lives in ../assets. A delivered file in assets/generated wins over the
 * code-drawn one in assets/interim at the same path (extension aside), one
 * file at a time. `virtual:art` maps each art name to the URL that won.
 * The same plugin serves onnxruntime's wasm from /ort.
 */
function artAndRuntime(): Plugin {
  const pick = (): Record<string, { url: string; file: string; source: 'generated' | 'interim' }> => {
    const out: Record<string, { url: string; file: string; source: 'generated' | 'interim' }> = {}
    for (const source of ['interim', 'generated'] as const) {
      for (const rel of walk(path.join(assets, source))) {
        if (!/\.(svg|png|webp|jpg)$/i.test(rel)) continue
        const name = rel.replace(/\.[^.]+$/, '').split(path.sep).join('/')
        out[name] = { url: `/art/${source}/${rel.split(path.sep).join('/')}`, file: path.join(assets, source, rel), source }
      }
    }
    return out
  }
  return {
    name: 'curbcut-art-and-runtime',
    resolveId: (id) => (id === 'virtual:art' ? '\0virtual:art' : null),
    load(id) {
      if (id !== '\0virtual:art') return null
      const urls = Object.fromEntries(Object.entries(pick()).map(([name, v]) => [name, { url: v.url, source: v.source }]))
      return `export default ${JSON.stringify(urls)}`
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0] ?? ''
        const file = url.startsWith('/art/') ? path.join(assets, url.slice(5)) : url.startsWith('/ort/') ? path.join(ortDist, path.basename(url)) : null
        if (!file || !existsSync(file) || !(file.startsWith(assets) || file.startsWith(ortDist))) return next()
        res.setHeader('Content-Type', MIME[path.extname(file)] ?? 'application/octet-stream')
        res.end(readFileSync(file))
      })
    },
    writeBundle(options) {
      const out = options.dir ?? 'dist'
      for (const { url, file } of Object.values(pick())) {
        const target = path.join(out, url)
        mkdirSync(path.dirname(target), { recursive: true })
        copyFileSync(file, target)
      }
      mkdirSync(path.join(out, 'ort'), { recursive: true })
      for (const name of ORT_FILES) copyFileSync(path.join(ortDist, name), path.join(out, 'ort', name))
    },
  }
}

const apiTarget = process.env.CURBCUT_API_URL ?? `http://localhost:${process.env.CURBCUT_API_PORT ?? 8109}`

export default defineConfig({
  plugins: [react(), artAndRuntime()],
  server: { proxy: { '/api': apiTarget } },
  preview: { proxy: { '/api': apiTarget } },
  build: { target: 'es2022', chunkSizeWarningLimit: 1600 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
