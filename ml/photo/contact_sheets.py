"""Lay the fetched photos out on numbered contact sheets for labelling by eye.

    .venv/bin/python -m ml.photo.contact_sheets

Writes data/photos/sheets/sheet-NN.jpg (git-ignored) and sheets.json, which
maps every cell number on every sheet to its manifest id. Photos are shuffled
with a fixed seed first, so the category a photo came from cannot be guessed
from its neighbours.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .fetch_commons import MANIFEST, PHOTO_DIR

OUT = PHOTO_DIR.parent / "sheets"
COLS, ROWS = 5, 4
CELL_W, CELL_H, LABEL_H = 256, 192, 18


def main() -> None:
    photos = json.loads(MANIFEST.read_text())["photos"]
    ids = [p["id"] for p in photos if (PHOTO_DIR / f"{p['id']}.jpg").exists()]
    random.Random(3).shuffle(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, int]] = {}
    per = COLS * ROWS
    for s in range(0, len(ids), per):
        sheet = Image.new("RGB", (COLS * CELL_W, ROWS * (CELL_H + LABEL_H)), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        name = f"sheet-{s // per + 1:02d}"
        index[name] = {}
        for i, pid in enumerate(ids[s : s + per]):
            try:
                img = ImageOps.exif_transpose(Image.open(PHOTO_DIR / f"{pid}.jpg")).convert("RGB")
            except Exception:  # noqa: BLE001 - an unreadable download is skipped, not fatal
                continue
            img.thumbnail((CELL_W - 4, CELL_H - 2))
            x = (i % COLS) * CELL_W + (CELL_W - img.width) // 2
            y = (i // COLS) * (CELL_H + LABEL_H) + LABEL_H + (CELL_H - img.height) // 2
            sheet.paste(img, (x, y))
            draw.text(((i % COLS) * CELL_W + 6, (i // COLS) * (CELL_H + LABEL_H) + 3), f"#{i + 1}", fill=(255, 220, 0))
            index[name][str(i + 1)] = pid
        sheet.save(OUT / f"{name}.jpg", quality=82)
    (OUT / "sheets.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} sheets for {len(ids)} photos in {OUT}")


if __name__ == "__main__":
    main()
