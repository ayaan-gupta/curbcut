# Curbcut: working notes for Claude Code

Project 9 of 11 from "Project PRDs: 11 Builds for Claude Code". Slug `curbcut`.

## Commands

- `make up` builds and runs everything in Docker with plain `docker` commands (this machine has no compose plugin): web 4109, API 8109, PostGIS 5509, all on 127.0.0.1. `make down` stops it.
- `make dev` runs PostGIS in Docker, the API with reload and the Vite dev server. `make setup` only installs.
- `make test` runs pytest (API and ML) with the store tests on both the memory store and PostGIS, then the web typecheck and vitest. `make test-nodb` skips the PostGIS half.
- `cd web && npm run demo` records `docs/demo.gif` and the stills against a running stack. It submits synthetic reports.
- `cd web && npm run art` renders the map icons at 24 px into `docs/icons-24px.png` and the PWA icons.
- `.venv/bin/python -m ml.roughness.train`, then `-m ml.roughness.evaluate`. `-m ml.roughness.calibrate` compares the simulator with real gait.
- `.venv/bin/python -m ml.photo.fetch_commons --download`, then `-m ml.photo.evaluate`. Training needs `pip install -r ml/requirements-train.txt` and `-m ml.photo.train`.
- `.venv/bin/python scripts/build_pilot.py --refresh` re-pulls the pilot area from Overpass and USGS.

## Shared conventions

- **Repo:** one repo per project, named by its slug, MIT license, README.md with a demo GIF and a one-command quickstart.
- **Default stack:** TypeScript with Vite or Next.js for web, Python (FastAPI) for ML and audio services, SQLite for local data and Postgres for hosted. This PRD overrides it: a React PWA with MapLibre, FastAPI, PostGIS.
- **Creative direction:** portfolio pieces that feel designed. Style bible here: bold high-contrast pictograms that stay legible at 24 px.
- **Image requests:** Claude Code does not generate images. Every image is requested with a numbered asset ID, the exact prompt, size, format, save path under `/assets/generated`, and what it is for, logged in `/assets/prompts.md`.
- **Quality bar:** typed code, at least 3 automated tests per milestone, no secrets in git (`.env.example` only), and an ARCHITECTURE.md of at most one page.
- **Definition of done:** runs from a clean clone with one command, passes all acceptance tests, recorded demo, README lists limitations and the next three ideas.

## How the image rule was applied here

The eleven design-time images (seven map icons, the app icon, three onboarding illustrations) were requested in `assets/prompts.md` without waiting. Hand-written SVG in `assets/interim` stands in. A delivered file in `assets/generated` at the same path and name wins, per file (`web/vite.config.ts`, the `virtual:art` module). The photo classifier's synthetic training scenes are drawn by `ml/photo/render_synth.py`, never by an image generator.

## Honesty rules that are enforced in code

- Anything synthetic is flagged: seeded reports, reports from the simulated sensor feed or a sample photo, and every feature built from one. The OSM draft leaves them out unless asked and then says DO NOT UPLOAD.
- `ml/photo/dataset.assert_real` refuses to evaluate on anything that is not a real manifest photo.
- The two acceptance tests that are not met are strict xfails (`ml/tests`). If a retrained model clears the bar they fail until the claim is updated.
- Photo labels are model labels. `labels.json` says so and the tests check that it says so.
- No code talks to the OpenStreetMap editing API. A test greps for it.

## Writing

No em dashes anywhere, in docs, comments or UI copy. README prose should sound like a person: varied sentences, no rule-of-three lists, no "it's not X, it's Y" constructions.

## Things that will bite

- `web/src/lib/roughness/features.ts` and `ml/roughness/features.py` must change together. The parity fixture fails on both sides otherwise. The same goes for `resizeBox` and Pillow's `Image.BOX`.
- The roughness model answers one window late because it looks at the next window. `Recorder.stop()` flushes the last one.
- Reports carry two times. `created_at` is when the phone observed it (speed check, recency). `received_at` is when the server got it (rate limits). Do not merge them: a batch upload would look like teleporting, or a backdated report would dodge the limit.
- Every published port binds to 127.0.0.1. Testing on a real phone needs HTTPS (sensors require a secure context), so use a tunnel instead of opening the port.
- `data/raw`, `data/real` and `data/photos` are git-ignored downloads. The committed manifests rebuild them.
