/**
 * Drives the real app in a headless browser, checks each milestone works end
 * to end in a browser, and records docs/demo.gif plus a few stills.
 *
 *   make dev    (or make up)
 *   npm run demo
 *
 * It submits reports, so run it against a database you do not mind adding
 * demo data to. Everything it submits comes from the simulated feed or a
 * bundled sample photo and is stored as synthetic.
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { chromium, type Page } from 'playwright'
import sharp from 'sharp'
import { cachedChromium } from './chromium'
import { encodeGif, type Frame } from './lib/gif'

const BASE = process.env.DEMO_URL ?? 'http://localhost:4109'
const WIDTH = 1000
const HEIGHT = 640
const docs = path.resolve(import.meta.dirname, '..', '..', 'docs')

async function main() {
  mkdirSync(docs, { recursive: true })
  const browser = await chromium.launch({ executablePath: cachedChromium() })
  const context = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1, acceptDownloads: true })
  const page = await context.newPage()
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const frames: Frame[] = []
  const shoot = async (delay: number) => {
    const { data } = await sharp(await page.screenshot({ type: 'png' })).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
    frames.push({ rgba: new Uint8Array(data), delay })
  }
  const hold = async (ms: number, every = 260) => {
    for (let t = 0; t < ms; t += every) {
      await shoot(every)
      await page.waitForTimeout(60)
    }
  }
  const click = async (testId: string, wait = 500) => {
    await page.click(`[data-testid="${testId}"]`)
    await hold(wait)
  }
  const still = (name: string) => page.screenshot({ path: path.join(docs, name) })
  const report: Record<string, unknown> = {}

  // Scene 1: first run.
  await page.goto(`${BASE}/?map=offline`, { waitUntil: 'networkidle' })
  await page.evaluate(() => localStorage.clear())
  await page.reload({ waitUntil: 'networkidle' })
  await page.waitForSelector('[data-testid="map"][data-ready="true"]')
  await hold(1500)
  await click('onboarding-done', 600)

  // Scene 2: a route for a manual wheelchair goes round the reported missing curb cut. A stroller does not have to.
  await click('demo-route', 300)
  await page.waitForSelector('[data-testid="route-result"][data-status="ok"]')
  await hold(2600)
  report.manual = await page.textContent('[data-testid="route-result"]')
  await still('screen-route.png')
  await click('profile-stroller', 2200)
  report.stroller = await page.textContent('[data-testid="route-result"]')
  await click('profile-manual_wheelchair', 900)
  await click('toggle-network', 2400)
  await still('screen-closed-segments.png')
  await click('toggle-network', 300)

  // Scene 3: passive mode on the simulated feed.
  await click('nav-record', 900)
  await click('record-start', 300)
  await page.waitForSelector('[data-testid="live-class"]')
  const seen = new Set<string>()
  for (let i = 0; i < 80 && !(await page.$('[data-testid="record-summary"]')); i++) {
    seen.add((await page.getAttribute('[data-testid="live-class"]', 'data-class').catch(() => '')) ?? '')
    await shoot(220)
    if (i === 34) await still('screen-record.png')
    await page.waitForTimeout(160)
  }
  await page.waitForSelector('[data-testid="record-summary"]')
  report.liveClasses = [...seen].filter(Boolean)
  report.recordSummary = await page.textContent('[data-testid="record-summary"]')
  await hold(1600)
  await click('record-upload', 300)
  await page.waitForSelector('[data-testid="record-message"]')
  report.recordMessage = await page.textContent('[data-testid="record-message"]')
  await hold(2000)

  // Scene 4: active mode. Faces are covered on the device, then a curb photo is classified and sent.
  await click('nav-photo', 900)
  await click('sample-faces', 200)
  await page.waitForSelector('[data-testid="photo-verdict"]', { timeout: 30000 })
  report.faces = await page.textContent('[data-testid="blur-summary"]')
  await hold(2200)
  await still('screen-face-blur.png')
  await page.click('text=Start over')
  await click('sample-curb', 200)
  await page.waitForSelector('[data-testid="photo-verdict"]', { timeout: 30000 })
  report.curbVerdict = await page.getAttribute('[data-testid="photo-verdict"]', 'data-label')
  await hold(1500)
  // Cover an area by hand, the way a licence plate would be.
  const box = (await page.locator('[data-testid="photo-canvas"]').boundingBox())!
  await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.12)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.3, { steps: 6 })
  await page.mouse.up()
  await hold(900)
  await page.locator('[data-testid="photo-pick"]').scrollIntoViewIfNeeded()
  await click('photo-pick', 500)
  const map = (await page.locator('[data-testid="map"]').boundingBox())!
  await page.mouse.click(map.x + map.width * 0.42, map.y + map.height * 0.55)
  await hold(700)
  await still('screen-photo.png')
  await page.locator('[data-testid="photo-submit"]').scrollIntoViewIfNeeded()
  await click('photo-submit', 300)
  await page.waitForSelector('[data-testid="photo-result"]')
  report.photoResult = await page.textContent('[data-testid="photo-result"]')
  await page.locator('[data-testid="photo-result"]').scrollIntoViewIfNeeded()
  await hold(2000)

  // Scene 5: review queue and the OpenStreetMap draft.
  await click('nav-review', 1400)
  report.heldReports = await page.locator('[data-testid="held-report"]').count()
  report.candidates = await page.locator('[data-testid="candidate"]').count()
  await page.locator('[data-testid="candidate"][data-kind="change"]').first().scrollIntoViewIfNeeded()
  await hold(1200)
  await page.locator('[data-testid="candidate"][data-kind="change"] [data-testid="candidate-approve"]').first().click()
  await hold(1200)
  await still('screen-review.png')
  await page.locator('[data-testid="download-osc"]').scrollIntoViewIfNeeded()
  await hold(500)
  const [empty] = await Promise.all([page.waitForEvent('download'), page.click('[data-testid="download-osc"]')])
  report.draftWithoutSynthetic = { file: empty.suggestedFilename(), note: await page.textContent('[data-testid="draft-note"]') }
  await hold(1500)
  await page.click('[data-testid="allow-synthetic"]')
  const [draft] = await Promise.all([page.waitForEvent('download'), page.click('[data-testid="download-osc"]')])
  const draftPath = path.join(docs, 'sample-draft-SYNTHETIC-do-not-upload.osc')
  await draft.saveAs(draftPath)
  report.draftWithSynthetic = { file: draft.suggestedFilename(), note: await page.textContent('[data-testid="draft-note"]') }
  await hold(2200)

  // Stills only: phone width, and the keyless remote basemap.
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(`${BASE}/?map=offline&onboarding=0`, { waitUntil: 'networkidle' })
  await page.waitForSelector('[data-testid="map"][data-ready="true"]')
  await page.click('[data-testid="demo-route"]')
  await page.waitForSelector('[data-testid="route-result"]')
  await page.waitForTimeout(1200)
  await still('screen-phone.png')
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.goto(`${BASE}/?onboarding=0`, { waitUntil: 'load' })
  const remote = await page.waitForSelector('[data-testid="map"][data-ready="true"]', { timeout: 20000 }).then(() => true).catch(() => false)
  if (remote) {
    await page.click('[data-testid="demo-route"]')
    await page.waitForSelector('[data-testid="route-result"]')
    await page.waitForTimeout(4000)
    await still('screen-map-openfreemap.png')
  }
  report.remoteBasemapLoaded = remote

  await browser.close()
  const gif = encodeGif(frames, WIDTH, HEIGHT)
  writeFileSync(path.join(docs, 'demo.gif'), gif)
  console.log(JSON.stringify({ frames: frames.length, gifMB: +(gif.length / 1024 / 1024).toFixed(2), pageErrors: errors, ...report }, null, 2))
  if (errors.length) process.exit(1)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
