"""Re-measure the committed roughness model. Trains nothing.

    .venv/bin/python -m ml.roughness.evaluate

Regenerates the held-out simulated walks from their seeds (SYNTHETIC data),
runs web/public/models/roughness.onnx on them, and prints macro F1. If the
real Luo 2020 slice has been fetched it is scored too.
"""

from __future__ import annotations

import json
from typing import Any

from . import real_luo
from .train import MODEL_DIR, evaluate_real, evaluate_sim, real_sets, sim_set

TARGET = 0.85


def measure() -> dict[str, Any]:
    model = (MODEL_DIR / "roughness.onnx").read_bytes()
    out: dict[str, Any] = {
        "simulated_heldout": evaluate_sim(model, sim_set("heldout")),
        "simulated_heldout_shifted": evaluate_sim(model, sim_set("heldout_shifted")),
    }
    if real_luo.available():
        real = real_sets()
        assert real is not None
        out["real_luo2020_participants_21_to_30"] = evaluate_real(model, real[1])
    return out


def main() -> None:
    results = measure()
    for name, m in results.items():
        verdict = "meets" if m["macro_f1"] >= TARGET else "below"
        kind = "SYNTHETIC" if name.startswith("simulated") else "real, two classes"
        print(f"{name:40s} [{kind}] n={m['n']} macro F1 {m['macro_f1']:.3f} ({verdict} the {TARGET:.2f} target)")
    print(json.dumps({k: v["per_class"] for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
