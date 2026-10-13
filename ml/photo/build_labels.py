"""Merge the per-sheet label files written during labelling into labels.json.

    .venv/bin/python -m ml.photo.build_labels labels-A.json labels-B.json

Each input maps sheet name and cell number to one of: present, missing,
damaged, blocked, exclude. data/photos/sheets/sheets.json (written by
contact_sheets.py) maps cells back to manifest ids.

WHO LABELLED: the build agent, an AI model, looking at numbered contact
sheets. Nobody verified the labels. They are model labels, not human ground
truth, and every accuracy number computed from them inherits that.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .dataset import CLASSES, LABELS, MANIFEST, ROOT

SHEETS = ROOT / "data" / "photos" / "sheets" / "sheets.json"


def main(paths: list[str]) -> None:
    index = json.loads(SHEETS.read_text())
    manifest_ids = {p["id"] for p in json.loads(MANIFEST.read_text())["photos"]}
    labels: dict[str, str] = {}
    for path in paths:
        for sheet, cells in json.loads(Path(path).read_text()).items():
            for cell, label in cells.items():
                if label not in (*CLASSES, "exclude"):
                    raise SystemExit(f"{path} {sheet} #{cell}: unknown label {label!r}")
                pid = index[sheet].get(cell)
                if pid is None:
                    continue
                if str(pid) in labels and labels[str(pid)] != label:
                    label = "exclude"  # two passes disagreed: not ground truth for anything
                labels[str(pid)] = label
    counts = Counter(labels.values())
    LABELS.write_text(
        json.dumps(
            {
                "labelled_by": "The build agent (an AI model) looking at 256 px thumbnails on contact sheets, opening the full photo when unsure. Not verified by a person. These are model labels, not human ground truth.",
                "instructions": "present: a usable curb ramp is visible. missing: a raised curb with no ramp in frame. damaged: broken curb, ramp or walking surface. blocked: a vehicle or object obstructs the sidewalk or ramp. exclude: anything else, or unsure. When in doubt, exclude.",
                "spot_check": "The lead build agent labelled two sheets (31 photos outside the calibration examples) independently: 26 matched, in 4 the labeller excluded a photo the checker would have labelled, and 1 was a present/missing disagreement.",
                "manifest_photos": len(manifest_ids),
                "labelled": len(labels),
                "not_reached": len(manifest_ids) - len(labels),
                "counts": dict(sorted(counts.items())),
                "labels": dict(sorted(labels.items(), key=lambda kv: int(kv[0]))),
            },
            indent=1,
        )
    )
    print(f"{len(labels)} of {len(manifest_ids)} manifest photos labelled: {dict(counts)}")


if __name__ == "__main__":
    main(sys.argv[1:])
