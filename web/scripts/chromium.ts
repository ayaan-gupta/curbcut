import { existsSync, readdirSync } from 'node:fs'
import { homedir } from 'node:os'
import path from 'node:path'

/** Playwright's own download may not match the cached revision, so point at whatever headless shell is cached. */
export function cachedChromium(): string | undefined {
  const cache = path.join(homedir(), 'Library', 'Caches', 'ms-playwright')
  if (!existsSync(cache)) return undefined
  for (const dir of readdirSync(cache).filter((d) => d.startsWith('chromium_headless_shell-')).sort().reverse()) {
    const bin = path.join(cache, dir, 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell')
    if (existsSync(bin)) return bin
  }
  return undefined
}
