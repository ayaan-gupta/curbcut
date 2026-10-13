/** Quick look at the app at desktop and phone width. Usage: tsx scripts/shots.ts <outdir> [tab] */
import path from 'node:path'
import { chromium } from 'playwright'
import { cachedChromium } from './chromium'

const BASE = process.env.DEMO_URL ?? 'http://localhost:4109'
const out = process.argv[2] ?? '.'
const tab = process.argv[3] ?? 'map'

const browser = await chromium.launch({ executablePath: cachedChromium() })
for (const [name, width, height] of [['desktop', 1280, 800], ['phone', 390, 844]] as const) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
  page.on('pageerror', (e) => console.log(`[${name} pageerror]`, e.message.slice(0, 300)))
  page.on('console', (m) => m.type() === 'error' && console.log(`[${name} console]`, m.text().slice(0, 300)))
  await page.goto(`${BASE}/?map=offline&onboarding=0&tab=${tab}`, { waitUntil: 'networkidle' })
  await page.waitForSelector('[data-testid="map"][data-ready="true"]', { timeout: 15000 })
  if (tab === 'map') {
    await page.click('[data-testid="demo-route"]')
    await page.waitForSelector('[data-testid="route-result"]')
    await page.waitForTimeout(1200)
  }
  await page.screenshot({ path: path.join(out, `shot-${tab}-${name}.png`) })
  await page.close()
}
await browser.close()
