"""Retrain the roughness classifier from windows recorded and labelled in the app.

    .venv/bin/python -m ml.roughness.retrain path/to/curbcut-windows.json [--out DIR] [--no-sim]

The app's collection mode (Record tab, "Label training data") exports a JSON
file like this:

    {"format": "curbcut-windows-v1", "rate_hz": 50, "window": 100,
     "windows": [{"label": "smooth", "session": "a1b2", "index": 0,
                  "acc": [[x, y, z], ...100 rows], "gyro": [[x, y, z], ...100 rows]}, ...]}

`session` is one continuous recording and `index` the window's position in
it, so relative and context features can be rebuilt.

With two or more sessions the script reports leave-one-session-out macro F1,
which is the honest version of "a held-out walk". With one session it holds
out the last 30 percent in time and says so. The final model is trained on
everything recorded, plus the simulated set unless --no-sim is given, and
written with a metrics file that states exactly what the numbers are.

This script exists so the PRD's real 200-window set can replace the simulator.
It has only ever been run on synthetic exports (see ml/tests).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from . import features as F
from . import simulator as S
from .train import MODEL_DIR, fit, onnx_predict, sim_set, to_onnx_bytes


def load_export(path: Path) -> dict[str, np.ndarray]:
    doc = json.loads(path.read_text())
    if doc.get("format") != "curbcut-windows-v1":
        raise ValueError("not a curbcut-windows-v1 export")
    if doc.get("rate_hz") != S.FS or doc.get("window") != S.WINDOW:
        raise ValueError(f"expected {S.FS} Hz windows of {S.WINDOW} samples")
    rows = sorted(doc["windows"], key=lambda w: (w["session"], w["index"]))
    sessions = sorted({w["session"] for w in rows})
    x = np.array([np.concatenate([np.asarray(w["acc"], dtype=np.float32), np.asarray(w["gyro"], dtype=np.float32)], axis=1) for w in rows])
    y = np.array([S.CLASSES.index(w["label"]) for w in rows])
    walk = np.array([sessions.index(w["session"]) for w in rows])
    return {"x": x, "y": y, "walk": walk, "f": F.with_context(F.extract_many(x), walk)}


def cross_validate(data: dict[str, np.ndarray], sim: dict[str, np.ndarray] | None) -> dict[str, Any]:
    sessions = np.unique(data["walk"])
    present = sorted(int(c) for c in np.unique(data["y"]))

    def train_and_score(train_mask: np.ndarray) -> np.ndarray:
        x, y = data["f"][train_mask], data["y"][train_mask]
        if sim is not None:
            x, y = np.concatenate([x, sim["f"]]), np.concatenate([y, sim["y"]])
        return fit(x, y, "none").predict(data["f"][~train_mask])

    if len(sessions) >= 2:
        pred = np.zeros_like(data["y"])
        for s in sessions:
            mask = data["walk"] != s
            if len(np.unique(data["y"][mask])) < 2 and sim is None:
                continue
            pred[~mask] = train_and_score(mask)
        return {
            "protocol": f"leave one session out over {len(sessions)} recorded sessions",
            "macro_f1": round(float(f1_score(data["y"], pred, labels=present, average="macro", zero_division=0)), 4),
            "n": int(len(pred)),
        }
    order = np.arange(len(data["y"]))
    mask = order < int(0.7 * len(order))
    pred = train_and_score(mask)
    return {
        "protocol": "single session: trained on the first 70 percent in time, scored on the last 30 percent. This is weaker than a held-out walk.",
        "macro_f1": round(float(f1_score(data["y"][~mask], pred, labels=present, average="macro", zero_division=0)), 4),
        "n": int(np.sum(~mask)),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    parser.add_argument("--no-sim", action="store_true", help="train on the recorded windows only")
    args = parser.parse_args(argv)

    data = load_export(args.export)
    sim = None if args.no_sim else sim_set("train")
    counts = {S.CLASSES[c]: int(n) for c, n in enumerate(np.bincount(data["y"], minlength=3))}
    result: dict[str, Any] = {"recorded_windows": counts, "sessions": int(len(np.unique(data["walk"]))), "mixed_with_simulated": sim is not None}
    result["cross_validation"] = cross_validate(data, sim)

    x, y = data["f"], data["y"]
    if sim is not None:
        x, y = np.concatenate([x, sim["f"]]), np.concatenate([y, sim["y"]])
    model = to_onnx_bytes(fit(x, y, "none"))
    labels, _ = onnx_predict(model, data["f"][:4])
    result["sanity_predictions"] = [S.CLASSES[int(v)] for v in labels]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "roughness.onnx").write_bytes(model)
    (args.out / "roughness-retrain-metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    if sum(counts.values()) < 200:
        print(f"note: {sum(counts.values())} recorded windows. The PRD asks for at least 200.")
    return result


if __name__ == "__main__":
    main()
