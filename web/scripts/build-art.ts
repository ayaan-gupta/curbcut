/**
 * Renders the code-drawn art for checking and for the PWA icons.
 *
 *   npm run art
 *
 * Writes docs/icons-24px.png: every map icon at its true 24 px size on the
 * map's background colours, and the same pixels enlarged six times, so
 * legibility at 24 px can be judged by eye. Also writes the PNG app icons.
 */
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import sharp from 'sharp'

const root = path.resolve(import.meta.dirname, '..', '..')
const iconDir = path.join(root, 'assets', 'interim', 'icons')

async function main() {
  const names = readdirSync(iconDir).filter((f) => f.endsWith('.svg') && f !== 'app.svg').sort()
  const cell = 24 * 6 + 16
  const composites: sharp.OverlayOptions[] = []
  for (const [i, name] of names.entries()) {
    const small = await sharp(readFileSync(path.join(iconDir, name)), { density: 72 }).resize(24, 24).png().toBuffer()
    const big = await sharp(small).resize(144, 144, { kernel: 'nearest' }).png().toBuffer()
    composites.push({ input: small, left: i * cell + 68, top: 14 })
    composites.push({ input: small, left: i * cell + 68, top: 62 })
    composites.push({ input: big, left: i * cell + 8, top: 104 })
  }
  const width = names.length * cell
  const strip = (color: string, top: number) => ({ input: { create: { width, height: 48, channels: 3 as const, background: color } }, left: 0, top })
  await sharp({ create: { width, height: 262, channels: 3, background: '#ffffff' } })
    .composite([strip('#f6f5f1', 2), strip('#2b2f36', 50), ...composites])
    .png()
    .toFile(path.join(root, 'docs', 'icons-24px.png'))
  const app = readFileSync(path.join(iconDir, 'app.svg'))
  for (const size of [180, 192, 512]) await sharp(app, { density: 600 }).resize(size, size).png().toFile(path.join(root, 'web', 'public', `icon-${size}.png`))
  console.log(`rendered ${names.length} icons at 24 px: docs/icons-24px.png`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
