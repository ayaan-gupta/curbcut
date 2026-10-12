"""Train the roughness classifier, evaluate it, and export ONNX for the browser.

    .venv/bin/python -m ml.roughness.train

Training data: simulated walks (SYNTHETIC, see simulator.py) and, when
data/real/luo2020 has been fetched, real gait windows from participants 1 to
20 of Luo et al. 2020 (smooth and rough only).

Evaluation, all through the exported ONNX file with float32 inputs:
- held-out simulated walks drawn with a different seed from the training ranges
- held-out simulated walks drawn from wider ranges than training saw
- real windows from participants 21 to 30, whom the model never saw

Writes web/public/models/roughness.onnx, roughness.json next to it,
ml/roughness/metrics.json and the parity fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

from . import features as F
from . import real_luo
from . import simulator as S

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "web" / "public" / "models"
METRICS = Path(__file__).resolve().parent / "metrics.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "feature_parity.json"

SIM_SETS: dict[str, tuple[int, int, S.Ranges]] = {
    # name: (seed, walks, ranges)
    "train": (100, 140, S.TRAIN_RANGES),
    "val": (200, 30, S.TRAIN_RANGES),
    "heldout": (300, 40, S.TRAIN_RANGES),
    "heldout_shifted": (400, 40, S.SHIFTED_RANGES),
}
PARAMS = {"n_estimators": 220, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8, "random_state": 7}


def sim_set(name: str) -> dict[str, np.ndarray]:
    seed, walks, ranges = SIM_SETS[name]
    data = S.make_dataset(seed, walks, ranges)
    data["f"] = F.with_context(F.extract_many(data["x"]), data["walk"])
    return data


def real_sets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    if not real_luo.available():
        return None
    data = real_luo.spliced_walks(real_luo.load(), seed=11)
    data["f"] = F.with_context(F.extract_many(data["x"]), data["walk"])
    train = np.isin(data["participant"], real_luo.TRAIN_PARTICIPANTS)
    return {k: v[train] for k, v in data.items()}, {k: v[~train] for k, v in data.items()}


WEIGHTINGS = ["none", "sqrt", "balanced"]


def class_weights(y: np.ndarray, weighting: str) -> np.ndarray:
    """Per-sample weights. 'balanced' equalises classes, 'sqrt' goes half way, 'none' leaves the data as it is."""
    balanced = compute_sample_weight("balanced", y)
    return {"none": np.ones(len(y)), "sqrt": np.sqrt(balanced), "balanced": balanced}[weighting]


def fit(x: np.ndarray, y: np.ndarray, weighting: str, extra_weight: np.ndarray | None = None) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(**PARAMS)
    weight = class_weights(y, weighting)
    model.fit(x, y, sample_weight=weight if extra_weight is None else weight * extra_weight)
    return model


def pick_weighting(train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> tuple[str, dict[str, float]]:
    """Chooses the class weighting on validation walks, which are neither trained on nor reported as a result."""
    scores: dict[str, float] = {}
    for weighting in WEIGHTINGS:
        pred = fit(train["f"], train["y"], weighting).predict(val["f"])
        scores[weighting] = round(float(f1_score(val["y"], pred, average="macro")), 4)
    return max(scores, key=lambda k: scores[k]), scores


def to_onnx_bytes(model: GradientBoostingClassifier) -> bytes:
    proto = to_onnx(
        model,
        np.zeros((1, len(F.ALL_FEATURE_NAMES)), dtype=np.float32),
        options={id(model): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    return proto.SerializeToString()


def onnx_predict(model_bytes: bytes, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    labels, probs = session.run(None, {session.get_inputs()[0].name: x.astype(np.float32)})
    return np.asarray(labels), np.asarray(probs)


def report(y: np.ndarray, pred: np.ndarray, classes: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": round(float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)), 4),
        "accuracy": round(float(np.mean(y == pred)), 4),
        "per_class": {
            S.CLASSES[c]: {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_rows_true_cols_pred": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def evaluate_sim(model_bytes: bytes, data: dict[str, np.ndarray]) -> dict[str, Any]:
    pred, _ = onnx_predict(model_bytes, data["f"])
    out = report(data["y"], pred, [0, 1, 2])
    out["by_mode"] = {name: report(data["y"][data["mode"] == m], pred[data["mode"] == m], [0, 1, 2]) for m, name in ((0, "walking"), (1, "rolling"))}
    out["by_placement"] = {
        name: {"macro_f1": report(data["y"][data["placement"] == i], pred[data["placement"] == i], [0, 1, 2])["macro_f1"], "n": int(np.sum(data["placement"] == i))}
        for i, name in enumerate(S.PLACEMENTS)
        if np.any(data["placement"] == i)
    }
    return out


def evaluate_real(model_bytes: bytes, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Two real classes. A real window predicted as bump is simply wrong; macro F1 is over smooth and rough."""
    pred, _ = onnx_predict(model_bytes, data["f"])
    out = report(data["y"], pred, [0, 1])
    out["predicted_bump_fraction"] = round(float(np.mean(pred == 2)), 4)
    out["by_sensor"] = {
        name: report(data["y"][data["sensor"] == i], pred[data["sensor"] == i], [0, 1])["macro_f1"] for i, name in enumerate(real_luo.SENSORS)
    }
    return out


def write_fixture(model_bytes: bytes, data: dict[str, np.ndarray]) -> None:
    """Eight consecutive windows of one simulated walk, treated as a walk of their own.

    Features are recomputed from the rounded windows exactly as stored, so the
    file is self-contained: a second implementation can be checked against it
    without knowing anything else about the walk.
    """
    walk = int(data["walk"][np.nonzero(data["y"] == 2)[0][0]])
    idx = np.nonzero(data["walk"] == walk)[0][:8]
    windows = np.round(data["x"][idx].astype(np.float64), 5).astype(np.float32)
    feats = F.with_context(F.extract_many(windows), np.zeros(len(windows), dtype=int))
    _, probs = onnx_predict(model_bytes, feats)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(
            {
                "note": "SYNTHETIC. Eight consecutive windows of one simulated walk. Used by the Python and TypeScript parity tests.",
                "feature_names": F.ALL_FEATURE_NAMES,
                "windows": [w.astype(np.float64).round(5).tolist() for w in windows],
                "features": [[float(v) for v in row] for row in feats],
                "probabilities": [[round(float(v), 6) for v in row] for row in probs],
                "labels": [int(data["y"][i]) for i in idx],
            }
        )
    )


def main() -> None:
    sets = {name: sim_set(name) for name in SIM_SETS}
    real = real_sets()
    print({name: np.bincount(d["y"], minlength=3).tolist() for name, d in sets.items()})

    weighting, val_scores = pick_weighting(sets["train"], sets["val"])
    print("validation macro F1 by class weighting:", val_scores, "->", weighting)
    sim_only = to_onnx_bytes(fit(sets["train"]["f"], sets["train"]["y"], weighting))
    metrics: dict[str, Any] = {
        "note": "Every number here was computed by running the exported ONNX model. 'simulated' sets are SYNTHETIC.",
        "classes": S.CLASSES,
        "features": F.ALL_FEATURE_NAMES,
        "params": PARAMS,
        "class_weighting": {"chosen": weighting, "validation_macro_f1": val_scores},
        "training_windows": {"simulated": int(len(sets["train"]["y"]))},
        "sim_only_model": {
            "simulated_heldout": evaluate_sim(sim_only, sets["heldout"]),
            "simulated_heldout_shifted": evaluate_sim(sim_only, sets["heldout_shifted"]),
        },
    }
    shipped = sim_only
    trained_with_real = False
    if real is not None:
        real_train, real_test = real
        metrics["training_windows"]["real_luo2020_participants_1_to_20"] = int(len(real_train["y"]))
        metrics["sim_only_model"]["real_luo2020_participants_21_to_30"] = evaluate_real(sim_only, real_test)
        # The shipped model sees the real smooth and rough windows too.
        x = np.concatenate([sets["train"]["f"], real_train["f"]])
        y = np.concatenate([sets["train"]["y"], real_train["y"]])
        shipped = to_onnx_bytes(fit(x, y, weighting))
        trained_with_real = True
        metrics["shipped_model"] = {
            "trained_on": "simulated walks plus real Luo 2020 participants 1 to 20",
            "simulated_heldout": evaluate_sim(shipped, sets["heldout"]),
            "simulated_heldout_shifted": evaluate_sim(shipped, sets["heldout_shifted"]),
            "real_luo2020_participants_21_to_30": evaluate_real(shipped, real_test),
        }
        real_only = to_onnx_bytes(fit(real_train["f"], real_train["y"], "balanced"))
        metrics["real_only_model_two_classes"] = {"real_luo2020_participants_21_to_30": evaluate_real(real_only, real_test)}
    else:
        metrics["shipped_model"] = {"trained_on": "simulated walks only (real set not fetched)", **metrics["sim_only_model"]}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "roughness.onnx").write_bytes(shipped)
    (MODEL_DIR / "roughness.json").write_text(
        json.dumps(
            {
                "classes": S.CLASSES,
                "features": F.ALL_FEATURE_NAMES,
                "rate_hz": S.FS,
                "window": S.WINDOW,
                "baseline_windows": F.BASELINE_WINDOWS,
                "min_baseline": F.MIN_BASELINE,
                "trained_with_real": trained_with_real,
                "synthetic_training_data": True,
            },
            indent=1,
        )
    )
    METRICS.write_text(json.dumps(metrics, indent=1))
    write_fixture(shipped, sets["heldout"])

    def line(name: str, m: dict[str, Any]) -> None:
        print(f"{name:58s} n={m['n']:5d} macro F1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f}")

    for model_name in ("sim_only_model", "shipped_model", "real_only_model_two_classes"):
        for set_name, m in metrics.get(model_name, {}).items():
            if isinstance(m, dict) and "macro_f1" in m:
                line(f"{model_name} / {set_name}", m)
                for sub in ("by_mode",):
                    for k, v in m.get(sub, {}).items():
                        line(f"    {k}", v)
    print(f"model: {(MODEL_DIR / 'roughness.onnx').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
