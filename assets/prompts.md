# Image requests

11 assets requested, 0 delivered. Code-drawn SVG stands in for every one of them, in `assets/interim`.

These are design-time assets. Save each file at the path given and the app picks it up on the next build: a file in `assets/generated` wins over the code-drawn one in `assets/interim` with the same path and name, one file at a time (the extension may differ, so a delivered PNG replaces an interim SVG). Nothing is blocked while they are outstanding.

The synthetic curb scenes used to augment classifier training are not in this list. They are drawn procedurally by `ml/photo/render_synth.py`, are labelled synthetic, and are never part of evaluation.

**Style bible:** bold high-contrast pictograms that stay legible at 24 px. Flat solid shapes, no gradients, no shadows, no outlines thinner than a tenth of the icon width, no text. One pictogram per icon in white on a solid coloured shape, with a white keyline round the shape so it reads on a light or dark map. The seven map icons differ in outer shape as well as colour, so they can be told apart without colour vision.

**Palette:** present `#0A7B45`, missing `#C4281C`, damaged `#B25A00`, blocked `#5B2A9D`, surface `#1D4E89`, slope `#1F2933`, ink `#111111`, signal yellow `#FFD60A`, paper `#F6F5F1`, white `#FFFFFF`.

Check any delivered map icon the way the interim ones were checked: `cd web && npm run art` writes `docs/icons-24px.png` with each icon at true 24 px on a light and a dark strip.

## CC-001 map icon: curb cut present

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid green circle (#0A7B45) with a thin white keyline. Inside, in solid white, the side profile of a sidewalk that slopes gently down to road level: a flat ledge on the left, a smooth ramp falling to the right, a short flat road at the bottom right. No wheelchair symbol, no text, no gradient, no shadow. Centred, the pictogram fills about 65 percent of the circle.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/curb-present.png`
- **Used for:** map marker and photo result row for a curb cut that is there and usable.

## CC-002 map icon: no curb cut

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid red rounded square (#C4281C) with a thin white keyline. Inside, in solid white, the side profile of a full-height curb: a flat ledge on the left that drops straight down in one square step to a lower flat road on the right. No text, no gradient, no shadow. The pictogram fills about 65 percent of the square.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/curb-missing.png`
- **Used for:** map marker and photo result row for a raised curb with no ramp. This is the barrier the wheelchair route goes around.

## CC-003 map icon: damaged ramp

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid burnt-orange upward-pointing triangle with slightly rounded corners (#B25A00) and a thin white keyline. Inside, one thick white zigzag crack running from upper right to lower left, like a break in concrete. No exclamation mark, no text, no gradient, no shadow.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/curb-damaged.png`
- **Used for:** map marker and photo result row for a broken curb ramp.

## CC-004 map icon: blocked ramp

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid purple octagon (#5B2A9D) with a thin white keyline. Inside, in solid white, a road barrier seen from the front: one thick horizontal bar on two short legs. No text, no gradient, no shadow.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/curb-blocked.png`
- **Used for:** map marker and photo result row for a ramp or sidewalk with something parked or standing on it.

## CC-005 map icon: rough surface

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid deep-blue square with generously rounded corners (#1D4E89) and a thin white keyline. Inside, one thick white saw-tooth line running left to right with three peaks, with rounded line ends. No text, no gradient, no shadow.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/surface-rough.png`
- **Used for:** map marker for a stretch that phones felt as rough.

## CC-006 map icon: bump

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid deep-blue square with generously rounded corners (#1D4E89) and a thin white keyline. Inside, one thick white line that runs flat, rises over a single round hump in the middle, and runs flat again. No text, no gradient, no shadow.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/surface-bump.png`
- **Used for:** map marker for a single jolt, such as a lip or a root heave.

## CC-007 map icon: steep slope

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram icon, high contrast, legible at 24 px. A solid charcoal diamond, a square turned 45 degrees (#1F2933), with a thin white keyline. Inside, a solid white right-angled wedge rising steeply from lower left to upper right. No percent sign, no text, no gradient, no shadow.
- **Spec:** 96 x 96 px, 1:1, transparent background, PNG.
- **Save to:** `assets/generated/icons/slope-steep.png`
- **Used for:** map marker for a slope someone measured as steep.

## CC-008 app icon

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat app icon, high contrast. A signal-yellow rounded square (#FFD60A) filled edge to edge. In solid black (#111111), the side profile of a sidewalk sloping down a curb ramp to road level, large and centred, filling about 70 percent of the width. No text, no gradient, no shadow, no border.
- **Spec:** 1024 x 1024 px, 1:1, opaque background, PNG.
- **Save to:** `assets/generated/icons/app.png`
- **Used for:** the onboarding sheet. To use it as the installed PWA icon as well, run `npm run art` after saving it.

## CC-009 onboarding illustration: stand back

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram illustration on a white rounded rectangle, black (#111111) with signal-yellow (#FFD60A) accents only. Side view. On the left a simple standing figure holds a phone at chest height. Two dotted yellow sight lines run from the phone to a curb ramp drawn in profile on the right. Below, a thin grey measuring line between the figure's feet and the ramp with a yellow pill at its middle, to show two to three metres of distance. No lettering except optionally "2-3 m" in the pill. No gradients, no shadows, no faces.
- **Spec:** 960 x 600 px, 8:5, opaque white background, PNG.
- **Save to:** `assets/generated/onboarding/step-1-stand-back.png`
- **Used for:** first of three photo instructions on the Photo tab.

## CC-010 onboarding illustration: frame the edge

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram illustration on a white rounded rectangle, black (#111111) with signal-yellow (#FFD60A) accents only. A phone held upright, seen from the front. On its screen, a curb ramp in profile where the sidewalk meets the road, sitting fully inside four yellow corner brackets. The whole ramp is inside the frame with a little margin. No text, no gradients, no shadows.
- **Spec:** 960 x 600 px, 8:5, opaque white background, PNG.
- **Save to:** `assets/generated/onboarding/step-2-frame-the-edge.png`
- **Used for:** second of three photo instructions on the Photo tab.

## CC-011 onboarding illustration: faces and plates are covered

- **Status:** requested (code-drawn interim in use)
- **Prompt:** Bold flat pictogram illustration on a white rounded rectangle, black (#111111) with signal-yellow (#FFD60A) and mid-grey accents only. On the left the head and shoulders of a simple figure whose face is covered by a coarse mosaic of yellow and grey squares. On the right the front of a simple car whose number plate is covered by a solid yellow bar. Nothing identifiable: no facial features, no plate characters. No text, no gradients, no shadows.
- **Spec:** 960 x 600 px, 8:5, opaque white background, PNG.
- **Save to:** `assets/generated/onboarding/step-3-faces-and-plates.png`
- **Used for:** third of three photo instructions on the Photo tab.
