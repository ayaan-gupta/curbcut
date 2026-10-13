"""Procedural curb scenes for training augmentation only.

EVERY IMAGE THIS FILE PRODUCES IS SYNTHETIC. They are drawn with rectangles,
polygons and noise, not photographed and not made by an image generator. The
PRD allows them for training on two conditions, both enforced in code: each
file is labelled synthetic (the filename starts with "synthetic_" and it
lives under data/photos/synthetic), and none is ever counted in evaluation
(evaluate.py refuses any path that is not a real manifest photo).

    .venv/bin/python -m ml.photo.render_synth --count 400 --sheet docs/synthetic-samples.jpg

What a scene contains: a road and a sidewalk meeting along a tilted curb line,
then one of four things at that edge.

    present   the curb drops to road level over a ramp, often with a tactile pad
    missing   the curb runs straight through, sometimes with a crosswalk aimed at it
    damaged   a curb or ramp with cracks, missing chunks and rubble
    blocked   a ramp or sidewalk with a car, cone, barrier, bin or scooter on it
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "photos" / "synthetic"
CLASSES = ["present", "missing", "damaged", "blocked"]
SIZE = 448  # drawn at 2x, saved at 224

Color = tuple[int, int, int]


def _jitter(rng: np.random.Generator, color: Color, amount: int = 14) -> Color:
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in color)  # type: ignore[return-value]


def _texture(rng: np.random.Generator, base: Color, grain: float, size: int = SIZE) -> Image.Image:
    """A surface: flat colour, coarse blotches, fine speckle."""
    coarse = rng.normal(0, grain * 0.6, (size // 16 + 1, size // 16 + 1))
    coarse = np.asarray(Image.fromarray(coarse.astype(np.float32)).resize((size, size), Image.BICUBIC))
    fine = rng.normal(0, grain, (size, size))
    field = (coarse + fine)[..., None]
    return Image.fromarray(np.clip(np.array(base, dtype=np.float32)[None, None, :] + field, 0, 255).astype(np.uint8))


def _along(p0: np.ndarray, d: np.ndarray, n: np.ndarray, s: float, t: float) -> tuple[float, float]:
    """A point s along the curb line and t across it (positive is the sidewalk side)."""
    q = p0 + d * s + n * t
    return float(q[0]), float(q[1])


def render(rng: np.random.Generator, label: str) -> Image.Image:
    assert label in CLASSES
    size = SIZE
    asphalt = _jitter(rng, (rng.choice([58, 70, 84]),) * 3, 8)
    walk_base = [(176, 172, 164), (196, 190, 178), (150, 148, 146), (168, 120, 98), (140, 132, 120)][int(rng.integers(0, 5))]
    sidewalk = _jitter(rng, walk_base, 10)

    # Curb line: through a point near the middle, tilted.
    angle = math.radians(rng.uniform(-28, 28))
    d = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-d[1], d[0]])
    if rng.random() < 0.5:  # which side of the picture is the sidewalk
        n = -n
    p0 = np.array([size / 2 + rng.uniform(-30, 30), size * rng.uniform(0.38, 0.62)])
    span = size * 1.2

    img = _texture(rng, asphalt, rng.uniform(5, 11))
    draw = ImageDraw.Draw(img)

    # Road paint on roughly half the scenes: crosswalk bars running away from the curb.
    crosswalk = rng.random() < (0.6 if label in ("present", "missing") else 0.3)
    centre = rng.uniform(-size * 0.18, size * 0.18)
    if crosswalk:
        paint = _jitter(rng, (225, 225, 215), 12)
        bar = rng.uniform(16, 28)
        gap = bar * rng.uniform(0.8, 1.4)
        for k in range(-4, 5):
            s0 = centre + k * (bar + gap)
            draw.polygon([_along(p0, d, n, s0, -8), _along(p0, d, n, s0 + bar, -8), _along(p0, d, n, s0 + bar, -span), _along(p0, d, n, s0, -span)], fill=paint)

    # Sidewalk half plane, textured, with joints.
    walk = _texture(rng, sidewalk, rng.uniform(3, 8))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon([_along(p0, d, n, -span, 0), _along(p0, d, n, span, 0), _along(p0, d, n, span, span), _along(p0, d, n, -span, span)], fill=255)
    img.paste(walk, (0, 0), mask)
    draw = ImageDraw.Draw(img)
    joint = tuple(int(c * 0.72) for c in sidewalk)
    slab = rng.uniform(40, 110)
    for k in range(-8, 9):
        draw.line([_along(p0, d, n, k * slab + centre * 0.3, 0), _along(p0, d, n, k * slab + centre * 0.3, span)], fill=joint, width=int(rng.integers(1, 4)))
    if rng.random() < 0.6:
        for k in range(1, 6):
            draw.line([_along(p0, d, n, -span, k * slab), _along(p0, d, n, span, k * slab)], fill=joint, width=int(rng.integers(1, 3)))

    # The curb: a top strip on the sidewalk side and a darker face on the road side.
    curb_top = _jitter(rng, [(200, 198, 190), (170, 168, 162), (214, 190, 60), (190, 70, 60)][int(rng.choice(4, p=[0.45, 0.35, 0.12, 0.08]))], 10)
    curb_face = tuple(int(c * rng.uniform(0.5, 0.7)) for c in curb_top)
    top_w = rng.uniform(10, 20)
    face_w = rng.uniform(10, 26)
    has_ramp = label == "present" or (label in ("damaged", "blocked") and rng.random() < 0.55)
    ramp_half = rng.uniform(45, 85)
    flare = rng.uniform(20, 45)

    def face_at(s: float) -> float:
        """Curb face height along the line: full, tapering through the flare, zero across the ramp."""
        if not has_ramp:
            return face_w
        dist = abs(s - centre)
        if dist <= ramp_half:
            return 0.0
        return face_w * min(1.0, (dist - ramp_half) / flare)

    step = 6
    for s in np.arange(-span, span, step):
        f0, f1 = face_at(s), face_at(s + step)
        draw.polygon([_along(p0, d, n, s, 0), _along(p0, d, n, s + step, 0), _along(p0, d, n, s + step, top_w), _along(p0, d, n, s, top_w)], fill=curb_top)
        if f0 > 0.5 or f1 > 0.5:
            draw.polygon([_along(p0, d, n, s, 0), _along(p0, d, n, s + step, 0), _along(p0, d, n, s + step, -f1), _along(p0, d, n, s, -f0)], fill=curb_face)

    if has_ramp:
        depth = rng.uniform(70, 130)
        ramp_color = tuple(int(c * rng.uniform(0.82, 0.95)) for c in sidewalk)
        draw.polygon(
            [_along(p0, d, n, centre - ramp_half, -2), _along(p0, d, n, centre + ramp_half, -2), _along(p0, d, n, centre + ramp_half * 0.9, depth), _along(p0, d, n, centre - ramp_half * 0.9, depth)],
            fill=ramp_color,
        )
        for side in (-1, 1):  # flares
            draw.polygon(
                [_along(p0, d, n, centre + side * ramp_half, -2), _along(p0, d, n, centre + side * (ramp_half + flare), 0), _along(p0, d, n, centre + side * ramp_half * 0.9, depth)],
                fill=tuple(int(c * 0.9) for c in ramp_color),
            )
        if rng.random() < 0.7:  # tactile pad with a dot grid
            pad = [(226, 186, 40), (176, 64, 50), (70, 70, 74), (230, 230, 220)][int(rng.integers(0, 4))]
            pw, pd0, pd1 = ramp_half * 0.8, rng.uniform(4, 14), rng.uniform(38, 60)
            draw.polygon([_along(p0, d, n, centre - pw, pd0), _along(p0, d, n, centre + pw, pd0), _along(p0, d, n, centre + pw, pd1), _along(p0, d, n, centre - pw, pd1)], fill=pad)
            dot = tuple(int(c * 0.78) for c in pad)
            for a in np.arange(-pw + 5, pw - 3, 9):
                for b in np.arange(pd0 + 5, pd1 - 3, 9):
                    x, y = _along(p0, d, n, centre + a, b)
                    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=dot)

    if label == "damaged":
        dark = tuple(int(c * 0.35) for c in asphalt)
        for _ in range(int(rng.integers(2, 6))):  # chunks missing from the curb or the walking surface
            s, t = centre + rng.uniform(-110, 110), rng.uniform(-face_w, 70)
            r = rng.uniform(10, 34)
            pts = [_along(p0, d, n, s + r * rng.uniform(0.5, 1.2) * math.cos(a), t + r * rng.uniform(0.4, 1.0) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
            draw.polygon(pts, fill=_jitter(rng, dark, 10))
        for _ in range(int(rng.integers(3, 8))):  # cracks
            s, t = centre + rng.uniform(-150, 150), rng.uniform(0, 140)
            pts = [_along(p0, d, n, s, t)]
            for _ in range(int(rng.integers(4, 9))):
                s, t = s + rng.uniform(-28, 28), t + rng.uniform(-26, 30)
                pts.append(_along(p0, d, n, s, t))
            draw.line(pts, fill=dark, width=int(rng.integers(2, 5)), joint="curve")
        for _ in range(int(rng.integers(15, 60))):  # rubble
            x, y = _along(p0, d, n, centre + rng.uniform(-130, 130), rng.uniform(-40, 90))
            r = rng.uniform(1.5, 5)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=_jitter(rng, sidewalk, 30))

    if label == "blocked":
        kind = str(rng.choice(["car", "car", "cone", "barrier", "bin", "scooter"]))
        s, t = centre + rng.uniform(-30, 30), rng.uniform(-10, 60)
        cx, cy = _along(p0, d, n, s, t)
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        if kind == "car":
            body = _jitter(rng, [(190, 30, 36), (30, 60, 140), (235, 235, 235), (40, 40, 44), (150, 150, 155), (20, 110, 80)][int(rng.integers(0, 6))], 10)
            w, h = rng.uniform(190, 300), rng.uniform(95, 135)
            ld.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=h * 0.28, fill=body + (255,))
            ld.rounded_rectangle([cx - w * 0.3, cy - h * 0.36, cx + w * 0.22, cy + h * 0.36], radius=h * 0.2, fill=(26, 32, 40, 255))
            ld.rounded_rectangle([cx - w * 0.2, cy - h * 0.28, cx + w * 0.12, cy + h * 0.28], radius=h * 0.15, fill=tuple(int(c * 0.85) for c in body) + (255,))
            for wx in (-0.3, 0.3):
                for wy in (-0.52, 0.52):
                    ld.ellipse([cx + wx * w - 20, cy + wy * h - 9, cx + wx * w + 20, cy + wy * h + 9], fill=(18, 18, 18, 255))
        elif kind == "cone":
            for k in range(int(rng.integers(1, 4))):
                ox = cx + (k - 1) * rng.uniform(40, 70)
                ld.polygon([(ox, cy - 56), (ox - 22, cy + 20), (ox + 22, cy + 20)], fill=(238, 110, 30, 255))
                ld.polygon([(ox - 9, cy - 24), (ox + 9, cy - 24), (ox + 13, cy - 8), (ox - 13, cy - 8)], fill=(245, 245, 240, 255))
                ld.rectangle([ox - 28, cy + 18, ox + 28, cy + 27], fill=(40, 40, 40, 255))
        elif kind == "barrier":
            w = rng.uniform(170, 260)
            ld.rectangle([cx - w / 2, cy - 30, cx + w / 2, cy + 6], fill=(240, 240, 235, 255))
            for k in np.arange(-w / 2, w / 2, 34):
                ld.polygon([(cx + k, cy - 30), (cx + k + 17, cy - 30), (cx + k + 5, cy + 6), (cx + k - 12, cy + 6)], fill=(210, 40, 36, 255))
            for leg in (-0.4, 0.4):
                ld.rectangle([cx + leg * w - 5, cy + 6, cx + leg * w + 5, cy + 50], fill=(60, 60, 64, 255))
        elif kind == "bin":
            for k in range(int(rng.integers(1, 3))):
                ox = cx + k * 80 - 30
                color = _jitter(rng, [(40, 110, 60), (60, 64, 70), (30, 80, 150)][int(rng.integers(0, 3))], 10)
                ld.rounded_rectangle([ox - 34, cy - 60, ox + 34, cy + 36], radius=8, fill=color + (255,))
                ld.rectangle([ox - 38, cy - 68, ox + 38, cy - 54], fill=tuple(int(c * 0.7) for c in color) + (255,))
        else:  # scooters parked in a row
            for k in range(int(rng.integers(2, 5))):
                ox = cx + (k - 1.5) * rng.uniform(46, 60)
                color = _jitter(rng, [(40, 44, 60), (170, 30, 40), (220, 220, 225)][int(rng.integers(0, 3))], 10)
                ld.rounded_rectangle([ox - 14, cy - 52, ox + 14, cy + 34], radius=10, fill=color + (255,))
                ld.ellipse([ox - 10, cy + 26, ox + 10, cy + 52], fill=(16, 16, 16, 255))
                ld.rectangle([ox - 22, cy - 58, ox + 22, cy - 50], fill=(30, 30, 30, 255))
        layer = layer.rotate(math.degrees(-angle) + rng.uniform(-18, 18), center=(cx, cy), resample=Image.BICUBIC)
        shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        shadow.paste((0, 0, 0, 90), (int(rng.uniform(6, 16)), int(rng.uniform(6, 16))), layer.split()[3])
        img = Image.alpha_composite(Image.alpha_composite(img.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(5))), layer).convert("RGB")

    # Light: a soft gradient, sometimes a hard cast shadow across the scene.
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size] / size
    ga = rng.uniform(0, 2 * math.pi)
    arr *= (rng.uniform(0.75, 1.15) + rng.uniform(0.0, 0.35) * (math.cos(ga) * (xx - 0.5) + math.sin(ga) * (yy - 0.5)))[..., None]
    if rng.random() < 0.35:
        shade = Image.new("L", (size, size), 255)
        sx = rng.uniform(0, size)
        ImageDraw.Draw(shade).polygon([(sx, 0), (sx + rng.uniform(60, 220), 0), (sx + rng.uniform(-200, 300), size), (sx - rng.uniform(40, 200), size)], fill=int(rng.uniform(110, 170)))
        arr *= (np.asarray(shade.filter(ImageFilter.GaussianBlur(rng.uniform(2, 14)))).astype(np.float32) / 255.0)[..., None]
    arr *= np.array([rng.uniform(0.92, 1.08), 1.0, rng.uniform(0.9, 1.1)], dtype=np.float32)
    arr += rng.normal(0, rng.uniform(2, 7), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    return out.resize((size // 2, size // 2), Image.LANCZOS)


def make_set(count_per_class: int, seed: int = 31, out: Path = OUT) -> list[tuple[Path, str]]:
    """Writes count_per_class images per class. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    written: list[tuple[Path, str]] = []
    for label in CLASSES:
        folder = out / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count_per_class):
            path = folder / f"synthetic_{label}_{i:04d}.jpg"
            render(rng, label).save(path, quality=int(rng.integers(70, 93)))
            written.append((path, label))
    return written


def sheet(path: Path, seed: int = 5) -> None:
    """A labelled sample sheet for the docs: four of each class, stamped SYNTHETIC."""
    rng = np.random.default_rng(seed)
    cell = 224
    canvas = Image.new("RGB", (4 * cell, 4 * cell + 28), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "SYNTHETIC training augmentation, drawn in code. Rows: present, missing, damaged, blocked. Never used in evaluation.", fill=(255, 214, 0))
    for row, label in enumerate(CLASSES):
        for col in range(4):
            canvas.paste(render(rng, label), (col * cell, 28 + row * cell))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400, help="images per class")
    parser.add_argument("--sheet", type=Path, default=None)
    args = parser.parse_args()
    written = make_set(args.count)
    print(f"wrote {len(written)} SYNTHETIC images under {OUT.relative_to(ROOT)}")
    if args.sheet:
        sheet(ROOT / args.sheet)
        print(f"sample sheet: {args.sheet}")


if __name__ == "__main__":
    main()
