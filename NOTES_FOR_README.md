# Notes for the README (Curbcut, project 9 of 11)

Plain facts for whoever writes the README. Every number here was measured in the build session by the command next to it.

## Quickstart, ports, env

- One command from a clean clone: `make up`. Needs Docker only. It builds two images and runs three containers with plain `docker` commands (this machine has the docker CLI on colima and no compose plugin). Then open http://localhost:4109.
- `make down` stops the containers and keeps the data volumes (`curbcut-db`, `curbcut-uploads`).
- Ports, all published on 127.0.0.1 only: web 4109, API 8109, PostGIS 5509. Override with `CURBCUT_WEB_PORT`, `CURBCUT_API_PORT`, `CURBCUT_DB_PORT`. `scripts/stack.sh` refuses to start on a port that is already listening.
- `docker-compose.yml` describes the same three services for machines that have compose. It was written but could not be run here.
- Development: `make dev` (PostGIS in Docker, uvicorn with reload, Vite). Tests: `make test`. Both create `.venv` (Python 3.12) and `web/node_modules` first.
- Env vars (all optional, all in `.env.example`; `scripts/stack.sh` sources `.env` if present):
  - `CURBCUT_WEB_PORT`, `CURBCUT_API_PORT`, `CURBCUT_DB_PORT`, `CURBCUT_DB_PASSWORD` (local default `curbcut`), `POSTGIS_IMAGE` (default `imresamu/postgis:17-3.5`, multi-arch)
  - `CURBCUT_DATABASE_URL` (empty means the in-memory store), `CURBCUT_SEED_DEMO` (default true), `CURBCUT_REVIEWER_TOKEN` (empty leaves review and export endpoints open), `CURBCUT_DEVICE_SALT`, `CURBCUT_UPLOAD_DIR`
  - `CURBCUT_PUBLISH_THRESHOLD` (0.5), `CURBCUT_EXPORT_THRESHOLD` (0.6), `CURBCUT_CLUSTER_RADIUS_M` (5), `CURBCUT_RATE_LIMIT_10MIN` (40), `CURBCUT_RATE_LIMIT_DAY` (400), `CURBCUT_MAX_ACCURACY_M` (25), `CURBCUT_MAX_SPEED_MPS` (12), `CURBCUT_MAX_DISTANCE_FROM_PATH_M` (25), `CURBCUT_CORS_ORIGINS`, `CURBCUT_MAX_PHOTO_BYTES`
  - `VITE_MAP_STYLE_URL` (build time; default the keyless OpenFreeMap Positron style; `offline` uses the built-in basemap). `?map=offline` in the address forces the built-in basemap at run time.
  - `CURBCUT_TEST_DATABASE_URL` (tests also run on PostGIS when set), `DEMO_URL` (demo recorder target)
- Testing on a real phone: browsers only give motion sensors and the camera to a secure context, so use an HTTPS tunnel to port 4109. No port was opened to the LAN and nothing in the repo does so.

## Tests

- From a fresh clone (`git clone`, then `make up`, then `make test`, in `scratchpad/clone-curbcut`): `make up` took 86 s and served the app on 4109 with the API seeded; pytest **109 passed, 1 skipped, 2 xfailed**; vitest **28 passed**; `tsc --noEmit` clean.
- The skip is the photo re-measurement, which needs the Commons photos on disk (they are not committed). With them present it passes. The 2 xfails are the two unmet acceptance tests.
- Store-backed API tests run twice, on the memory store and on PostGIS. Without `CURBCUT_TEST_DATABASE_URL` the PostGIS half is skipped (`make test-nodb`).
- By milestone: M1 `api/tests/test_m1_tagged_points.py` (7) and `web/tests/m1-trace.test.ts` (5). M2 `ml/tests/test_roughness.py` (10) and `web/tests/m2-roughness.test.ts` (14). M3 `ml/tests/test_photo.py` (7) and `web/tests/m3-photo.test.ts` (9). M4 `api/tests/test_m4_*.py` and `test_seed.py` (44).
- `cd web && npm run demo` is the browser end-to-end check: it drives every tab against a running stack and fails on any page error. It ran clean against the Docker build.

## What each milestone delivered

- **M1, GPS trace recording and a map of tagged points.** `Recorder` (`web/src/lib/sensors`) takes Geolocation fixes and DeviceMotion samples behind explicit permission prompts, only between Start and Stop, and stops if the page is hidden. The trace stays on the device; tagged points go to `POST /api/reports/batch`. MapLibre map with the feature layer, a keyless remote style and a built-in offline basemap drawn from the committed OSM extract. Real pilot area: Capitol Hill, Seattle, 991 footway edges, 21 km, 256 kerb nodes, pulled from Overpass, with incline from the USGS 3DEP 1 m lidar DEM.
- **M2, roughness classifier with offline evaluation.** 50 Hz, 2 s windows, 21 features (vertical RMS, jerk, four spectral bands and more), plus features relative to the walk's running median and from the neighbouring windows. Gradient-boosted trees exported to ONNX (372 kB), run in the browser by onnxruntime-web. A simulator of walking and rolling, calibrated against real gait recordings, training and evaluation scripts, the in-app collection mode that exports labelled windows, and `ml/roughness/retrain.py` that retrains from such an export.
- **M3, photo capture, on-device classification, privacy blurring.** MobileNetV3-Small fine-tuned to four classes, ONNX (6.1 MB) in the browser. UltraFace RFB-320 (MIT, 1.3 MB) finds faces in the browser, with a tiled second pass for small faces, and they are replaced by a three-cell mosaic before the photo is shown or sent. A drag tool covers anything else by hand. The server strips EXIF as a second check. Real-photo dataset pipeline with a per-file licence manifest.
- **M4, aggregation, confidence, routing, OSM draft.** Reports cluster within 5 m per category; confidence from independent reporters, agreement and recency; publish at 0.5. Device rate limits, outlier rejection, a review queue. Three routing profiles as hard filters on slope, surface and kerb height, with the shortest walk shown alongside and an honest "no accessible route". An OsmChange draft built only from edits a reviewer approved, a review screen, and a GeoJSON dataset export under ODbL 1.0.

## Acceptance tests

| PRD acceptance test | Status | Measured | Where |
|---|---|---|---|
| Roughness classifier at least 85% macro F1 on a held-out walk | **Not met.** Evaluated on synthetic data, and below target there too | Macro F1 **0.833** on 40 held-out **simulated** walks (3,600 windows, different seed from training). 0.829 on 40 simulated walks drawn from wider parameter ranges than training. No real labelled phone walk exists | `.venv/bin/python -m ml.roughness.evaluate`; `ml/roughness/metrics.json`; `ml/tests/test_roughness.py` (strict xfail on the 0.85 claim, regression floor at 0.80) |
| Photo classifier at least 85% accuracy on a real-photo test set with no synthetic images | **Not met** | **70.0%** on 80 real photos (56 of 80, 95% Wilson interval 59% to 79%). Majority-class baseline 47.5%. Labels are model labels, not human ground truth | `.venv/bin/python -m ml.photo.evaluate` (after `-m ml.photo.fetch_commons --download`); `ml/photo/metrics.json`; `ml/tests/test_photo.py` (strict xfail) |
| Two reports within 5 m merge, and a lone low-confidence report does not publish | **Met**, automated, on the memory store and on PostGIS | Two reports 3.9 m apart become one feature with two reporters; 8 m apart stay separate. A lone report at model confidence 0.3 scores under 0.5 and is not published. A lone report at model confidence 1.0 scores 0.39 and is not published either | `api/tests/test_m4_aggregation.py` |
| Wheelchair route never crosses a segment marked "no curb cut" when an alternative exists | **Met**, automated, including the no-alternative case | Small street: the route goes round by the other crossing (120 m against 20 m). With both crossings marked, the answer is `no_accessible_route` with no route geometry. On the real pilot graph, 300 random trips with a fifth of the crossings marked: every returned route avoids all marked segments, and "route returned" equals "alternative exists" as decided by an independent flood fill | `api/tests/test_m4_routing.py` |

Two of four are met. The two model targets are missed, and the tests that encode them are strict xfails: the suite is green, the miss is visible in the test report, and if a later model clears the bar the xfail fails until the claim is updated.

### Roughness numbers in full (all from `ml/roughness/metrics.json`, produced by `python -m ml.roughness.train`)

- Training set: 12,600 simulated windows from 140 simulated walks, plus 4,890 real windows (smooth and rough only) from participants 1 to 20 of Luo et al. 2020.
- Shipped model, held-out simulated walks: macro F1 0.833, accuracy 0.869. Per class F1: smooth 0.92, rough 0.82, bump 0.75. Walking personas alone 0.786, rolling personas alone 0.880.
- Shipped model, real held-out people (Luo participants 21 to 30, 2,496 windows, two classes only): macro F1 **0.712** over smooth and rough. By sensor: lower back 0.74, wrist 0.68, thigh 0.70.
- Same architecture trained on simulation only: 0.846 on held-out simulation, **0.628** on the real held-out people. That gap is the sim-to-real cost.
- Trained on the real windows only (two classes): 0.733 on the real held-out people. So about 0.73 is the ceiling for this feature set on that data.
- The class weighting was chosen on 30 separate validation walks, which are neither trained on nor reported.

### Photo numbers in full (`ml/photo/metrics.json`, produced by `python -m ml.photo.train`)

- 884 photos fetched from Wikimedia Commons, 780 labelled, 226 kept in the four classes (present 61, missing 48, damaged 38, blocked 79), 554 excluded as not showing a curb state. 104 were never reached.
- Split by author plus Commons category (130 groups) so a photographer's series stays on one side: train 119, validation 27, test 80 (present 14, missing 17, damaged 11, blocked 38).
- Real only: validation 77.8%, test 73.8%. Real plus 1,200 synthetic scenes: validation 81.5%, test 70.0%. Synthetic only: test 22.5%, which is below always guessing "blocked".
- The shipped model is real plus synthetic, because the rule fixed before testing was "best on validation". With 27 validation photos that choice is close to a coin flip, and on the test photos the real-only model was 3 photos better.
- Where it fails: 9 of 17 "missing" test photos are called "present". Blocked is 36 of 38.

## Where every photo and recording came from

- **Map data.** OpenStreetMap via the Overpass API, ODbL 1.0, fetched 2026-10-12. Contributor names, uids, changesets and timestamps were removed. Elevation: USGS 3DEP point query service, public domain. Committed as `data/pilot/capitol-hill.json` (551 kB) by `scripts/build_pilot.py`.
- **Real gait recordings.** Luo, Coppola, Dixon, Li, Dennerlein, Hu (2020), "A database of human gait performance on irregular and uneven surfaces collected by wearable sensors", Scientific Data 7:219, data at https://doi.org/10.6084/m9.figshare.c.4892463, **CC0**. Thirty people, six Xsens IMUs at 100 Hz. Used: lower back, wrist and right thigh sensors on "flat even" (as smooth) and "cobble stone" (as rough), 1,080 trials, 268 minutes. These are research IMUs strapped to the body, **not phones**, nobody is on wheels, and there is no bump class. Not committed; `ml/roughness/fetch_luo2020.py` fetches the slice again.
- **Simulated walks.** `ml/roughness/simulator.py`. Synthetic. Used for training, for the headline macro F1, for the app's demo sensor feed (`web/public/demo/walk.json`) and for test fixtures. Labelled synthetic in each of those places.
- **Real photos.** 884 files from Wikimedia Commons categories about curbs, kerb ramps, damaged sidewalks, sidewalk parking and illegal parking. Licences: CC BY-SA (669), CC BY (137), CC0 (51), public domain (26), no known restrictions (1). `ml/photo/manifest.json` records the source page, author, licence and SHA-256 of every file. The photos themselves are not committed; `python -m ml.photo.fetch_commons --download` fetches them.
- **Photo labels are model labels.** They were assigned by the build agent (an AI model) and two AI sub-agents it launched, looking at 256 px thumbnails on contact sheets. No person verified them. The sub-agents were cut off before finishing, which is why 104 photos have no label. Spot check: the lead agent labelled 31 photos independently and matched 26; in 4 the labeller had excluded a photo the checker would have kept; 1 was a present against missing disagreement. The 70.0% is agreement with those model labels and nothing stronger.
- **Synthetic curb scenes.** `ml/photo/render_synth.py`, drawn with polygons and noise. Training only. `docs/synthetic-samples.jpg` shows sixteen. No AI image generator was used anywhere.
- **Bundled sample photos** (`web/public/samples`, credits in the folder): a CC0 curb cut photo by BubbaJuice from Commons, which is in the training split, and NASA's public domain Apollo 11 crew portrait for the face blur. The same portrait is the face-detector test fixture.
- **Models from elsewhere.** Face detector: Ultra-Light-Fast-Generic-Face-Detector-1MB RFB-320 by Linzaer, MIT, from the ONNX Model Zoo, checksum pinned in `scripts/prepare_face_model.py`. Photo backbone: torchvision MobileNetV3-Small ImageNet weights, BSD-3-Clause.
- **Demo reports on the map.** All synthetic, from `api/curbcut/seed.py`, placed on real kerb nodes.

## Technical problems worth telling

1. **The first simulator was too clean, and calibrating it cost the target.** I compared simulated walkers against the real lower-back sensor feature by feature (`python -m ml.roughness.calibrate`). The simulated 3 to 8 Hz band had about 28 times less power than real gait (log10 power 0.28 against 1.72), jerk RMS was 35 against 83 m/s^3, crest factor 2.0 against 3.05. Real walking is dominated by heel strike with broadband content to 20 Hz; mine was nearly a sine wave. After three passes (heel strike made the dominant term, a second sharper ring, push-off, soft-tissue noise, left and right asymmetry) the medians are within about 20% for lower back and wrist. The same data showed that real cobblestone raises vertical RMS by only 17% over flat paving, so I set the simulated rough effect for walkers to match at its mild end. A classifier on that honest simulator scored 0.796.
2. **Context got five points back, and the comparison is not pure.** People differ from each other far more than rough differs from smooth under one person, so I added features relative to the walk's running median, plus six features from the window before and after (a jolt with calm neighbours is a bump, with noisy neighbours it is rough ground). That and choosing class weights on validation walks took the simulation-only model from 0.796 to 0.846. In the same change I also made the simulator's class definitions cleaner (a bump is a discontinuity of at least 20 mm, slab edges inside "rough" at most 7 mm times the roughness scale, and a bump is labelled in the window where its ringing shows), so part of that gain is an easier task and not a better model.
3. **Reading 5% of a 2 GB zip.** The real gait set is one 2.04 GB archive. `fetch_luo2020.py` gives Python's `zipfile` a file object backed by HTTP range requests, reads the central directory (1.4 MB), and pulls only the 1,080 members it needs: 138 MB.
4. **A walk uploaded in one batch looked like teleporting.** Every point in a batch got the server's clock, so twelve points 20 m apart arrived "in the same second" and the speed check held eleven of them for review. Reports now carry two times: when the phone observed it (speed check, recency) and when the server received it (rate limits). The observed time is clamped to the last 24 hours and never the future, and the rate limit ignores it, so backdating cannot dodge the limit. There is a test for each half.
5. **A blurred face was still a face to the detector.** My first mosaic was four cells across. On the test portrait the detector still fired on one of the three faces afterwards. At three cells across it finds none, and the test now asserts exactly that: blur, run the detector again, expect nothing.
6. **Synthetic scenes did not transfer.** Trained on procedural scenes alone the photo model scores 22.5% on real photos, under the 47.5% majority baseline. Added to real photos they moved validation up 4 points and test down 4, on sets of 27 and 80 photos, which is noise in both directions.
7. **GPS jitter tripled the distance.** The record screen said 415 m for a 149 m simulated roll, because 2 m of fix noise at 1 Hz adds up. Distance now only counts a fix once it is 8 m from the last one counted: 155 m.

## Surprises and things I got wrong first

- `docker compose` is not installed on this machine, although the brief said it was. The one command became `make up` over plain `docker` commands. The compose file is untested.
- Postgres was first published on all interfaces. It now binds to 127.0.0.1, as do the API and web ports.
- 71% of the fetched Commons photos were excluded during labelling. Category names like "Kerbs in London" mostly hold street scenes where the curb is incidental.
- The single-link confidence formula I started with let one perfect report reach 0.57. K was raised to 2 so one vote tops out at 0.39.
- I wrote a made-up checksum constant into `prepare_face_model.py` before having downloaded the file, noticed, and replaced it with the measured one. The script now refuses a file that does not match.
- A hand-made test signal (a still phone with a 13 m/s^2 ringing jolt) was classified smooth by the roughness model. It is outside anything the simulator produces (near-zero baseline noise, zero rotation). The test now uses a stub model for the plumbing; the brittleness is real and listed below.

## Limitations, bluntly

- Neither model meets its target, and the roughness figure is on simulated data. No phone has recorded a labelled walk for this project. The app has not been run on a physical phone at all: sensors, camera and permission prompts were exercised only through the simulated feed and bundled photos in headless Chromium.
- For someone walking, the roughness signal is weak in reality: 0.71 macro F1 on real smooth against cobblestone. The classifier is much more believable on wheels, and that half has no real data behind it.
- The roughness model is brittle outside the simulator's range, as the hand-made signal showed.
- Photo labels are model labels on a small set, and the classifier calls most raised curbs "present". It has no "not a curb" class, so any photo gets one of four answers (the crew portrait comes out "damaged").
- Licence plates are not detected. There is a manual cover tool and the app says so. Faces smaller than about 12 px in the tiled pass can be missed.
- Device identity is a random id made on the phone. Rate limits stop honest clients and do nothing against someone minting ids, so "independent reporters" is only as good as that.
- Review and export endpoints are open unless `CURBCUT_REVIEWER_TOKEN` is set. There are no user accounts.
- Clustering depends on arrival order and never merges two features that drift together. One process-wide lock serialises intake.
- GPS in a street canyon is often worse than the 5 m clustering radius. Accuracy over 25 m is rejected and anything over 8 m is down-weighted, but two honest reports of one curb can still land in two clusters.
- Incline exists for 355 of 991 edges (those with a 12 m stretch to measure over). Unknown incline is treated as passable.
- The OSM draft only modifies tags on existing elements. It cannot add a kerb node, and a surface edit applies to a whole way.
- Photo reports need a connection. There is no offline upload queue. The service worker caches the app shell and models only.
- The CI workflow was written and not run. `docker-compose.yml` likewise.
- One pilot area. Routing knows nothing outside it.

## Next three ideas

1. Collect the real 200-window set with the collection mode on a wheelchair and a stroller, retrain with `retrain.py`, and report leave-one-session-out macro F1, which is the acceptance test as the PRD meant it.
2. Have people label the photo set (the labelling sheets and manifest are ready), add a "not a curb" class, and fine-tune on photos taken the way the app asks for them.
3. Replace the made-up device id with platform attestation or a signed-in contributor, so "independent reporters" means something, and move intake locking into Postgres.

## What is real and what is a stand-in

- Real: OSM pilot graph and lidar incline; PostGIS store; clustering, scoring, abuse checks, routing, OSM draft; all three ONNX models running in the browser; face detector weights; the real gait slice; the Commons photos.
- Stand-ins, labelled as such in the UI and data: the simulated sensor feed used when a device has no motion sensors (it runs through the same pipeline as real sensors); the seeded demo reports; sample photos; the in-memory store used when no database URL is set; the built-in offline basemap when the tile host is unreachable.
- Interim art: seven map icons, the app icon and three onboarding illustrations are hand-written SVG in `assets/interim`. Requests for final versions are in `assets/prompts.md` (CC-001 to CC-011). Icons were rendered at 24 px and checked (`docs/icons-24px.png`).
- No LLM, no image generation, no SMS, no external API at run time other than map tiles.

## Files in docs/

- `demo.gif` (2.7 MB): the real app in headless Chromium against the Docker stack. Onboarding, the demo walk routed for a manual chair and a stroller, closed segments, a simulated roll being classified live and uploaded, face blur, a curb photo classified and sent, the review queue and the OSM draft.
- `screen-route.png`: the wheelchair route (yellow) going 76 m round a reported missing curb cut, shortest walk dashed.
- `screen-closed-segments.png`: every footway edge closed to a manual wheelchair, in red.
- `screen-record.png`: recording, with the live class, the acceleration trace and the tagged points on the map.
- `screen-face-blur.png`: three faces covered on the device.
- `screen-photo.png`: a classified curb photo with a hand-covered area, placed on the map.
- `screen-review.png`: OSM draft candidates, one approved.
- `screen-phone.png`: the map tab at 390 px wide.
- `screen-map-openfreemap.png`: the same route on the keyless OpenFreeMap basemap.
- `icons-24px.png`: the seven map icons at true 24 px on light and dark, and enlarged.
- `synthetic-samples.jpg`: sixteen procedural training scenes, stamped synthetic.
- `sample-draft-SYNTHETIC-do-not-upload.osc`: what the OsmChange draft looks like, built from seeded demo data and marked accordingly.
