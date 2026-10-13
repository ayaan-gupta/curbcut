"""The real-photo set: manifest, labels, and the train/validation/test split.

Labels were assigned by the build agent (an AI model looking at each photo),
not by a person. That is stated wherever a number from this set appears.

The split is by group, not by photo. A group is one author's photos from one
Commons category, because those tend to be a series of the same street taken
minutes apart. Keeping a whole series on one side stops near-duplicates from
leaking a test photo into training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "manifest.json"
LABELS = HERE / "labels.json"
PHOTO_DIR = ROOT / "data" / "photos" / "commons"
SYNTH_DIR = ROOT / "data" / "photos" / "synthetic"
CLASSES = ["present", "missing", "damaged", "blocked"]
INPUT = 224
TEST_SHARE = 0.4
VAL_SHARE = 0.15


@dataclass(frozen=True)
class Photo:
    id: int
    label: str
    group: str
    split: str  # train, val or test

    @property
    def path(self) -> Path:
        return PHOTO_DIR / f"{self.id}.jpg"


def _bucket(group: str) -> float:
    """A stable number in [0, 1) per group, so the split does not depend on file order or a random seed."""
    return int(hashlib.sha256(f"curbcut-split-v1:{group}".encode()).hexdigest()[:8], 16) / 0x100000000


def load_photos() -> list[Photo]:
    manifest = {p["id"]: p for p in json.loads(MANIFEST.read_text())["photos"]}
    labels = json.loads(LABELS.read_text())["labels"]
    photos: list[Photo] = []
    for key, label in sorted(labels.items(), key=lambda kv: int(kv[0])):
        if label not in CLASSES:
            continue
        meta = manifest[int(key)]
        group = f"{meta['author']}|{meta['found_in']}"
        b = _bucket(group)
        split = "test" if b < TEST_SHARE else "val" if b < TEST_SHARE + VAL_SHARE else "train"
        photos.append(Photo(id=int(key), label=label, group=group, split=split))
    return photos


def counts(photos: list[Photo]) -> dict[str, dict[str, int]]:
    out = {split: {c: 0 for c in CLASSES} for split in ("train", "val", "test")}
    for p in photos:
        out[p.split][p.label] += 1
    return out


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def deploy_preprocess(img: Image.Image) -> np.ndarray:
    """Exactly what the phone does: shorter side to 224 by area averaging, centre crop, RGB 0..255, CHW float32.

    web/src/lib/photo/pixels.ts implements the same resize.
    """
    w, h = img.size
    scale = INPUT / min(w, h)
    nw, nh = max(INPUT, int(w * scale + 0.5)), max(INPUT, int(h * scale + 0.5))
    img = img.resize((nw, nh), Image.BOX)
    left, top = (nw - INPUT) // 2, (nh - INPUT) // 2
    arr = np.asarray(img.crop((left, top, left + INPUT, top + INPUT)), dtype=np.float32)
    return arr.transpose(2, 0, 1)


def assert_real(path: Path) -> None:
    """Evaluation may only ever read real manifest photos."""
    resolved = path.resolve()
    if SYNTH_DIR.resolve() in resolved.parents or resolved.name.startswith("synthetic_") or PHOTO_DIR.resolve() not in resolved.parents:
        raise AssertionError(f"{path} is not a real manifest photo and must not be evaluated on")
