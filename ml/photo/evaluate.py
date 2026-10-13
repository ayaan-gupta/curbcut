"""Re-measure the committed photo classifier on the real test photos. Trains nothing.

    .venv/bin/python -m ml.photo.fetch_commons --download   # once: fetch the photos the manifest lists
    .venv/bin/python -m ml.photo.evaluate

The test set is real Wikimedia Commons photos only. assert_real() refuses
anything synthetic. The labels are model labels (see labels.json), not human
ground truth. Needs only onnxruntime and Pillow, not torch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .dataset import CLASSES, ROOT, Photo, assert_real, deploy_preprocess, load_image, load_photos

MODEL = ROOT / "web" / "public" / "models" / "curb-photo.onnx"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_probs.json"
SAMPLE = ROOT / "web" / "public" / "samples" / "curb-cut.jpg"
TARGET = 0.85


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def evaluate(photos: list[Photo], model: Path = MODEL) -> dict[str, Any]:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for p in photos:
        assert_real(p.path)
        probs = session.run(None, {"pixels": deploy_preprocess(load_image(p.path))[None]})[0][0]
        confusion[CLASSES.index(p.label), int(np.argmax(probs))] += 1
    n, correct = int(confusion.sum()), int(np.trace(confusion))
    return {"n": n, "accuracy": round(correct / max(1, n), 4), "accuracy_95ci_wilson": wilson(correct, n), "confusion_rows_true_cols_pred": confusion.tolist()}


def write_fixture() -> None:
    """Probabilities for the bundled sample photo, so the TypeScript pipeline can be checked against this one."""
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    probs = session.run(None, {"pixels": deploy_preprocess(load_image(SAMPLE))[None]})[0][0]
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"image": "web/public/samples/curb-cut.jpg", "classes": CLASSES, "probs": [round(float(v), 5) for v in probs]}))


def main() -> None:
    test = [p for p in load_photos() if p.split == "test"]
    present = [p for p in test if p.path.exists()]
    if len(present) < len(test):
        raise SystemExit(f"{len(test) - len(present)} of {len(test)} test photos are not on disk. Run: python -m ml.photo.fetch_commons --download")
    result = evaluate(present)
    verdict = "meets" if result["accuracy"] >= TARGET else "below"
    print(f"real-photo test accuracy {result['accuracy']:.3f} on {result['n']} photos, 95% CI {result['accuracy_95ci_wilson']} ({verdict} the {TARGET:.2f} target)")
    print("labels are model labels, not human ground truth. confusion (rows true, cols predicted,", CLASSES, "):", result["confusion_rows_true_cols_pred"])
    write_fixture()


if __name__ == "__main__":
    main()
