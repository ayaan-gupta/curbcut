"""Milestone 2: the roughness classifier and its offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from ml.roughness import features as F
from ml.roughness import retrain
from ml.roughness import simulator as S
from ml.roughness.evaluate import TARGET, measure
from ml.roughness.train import METRICS, MODEL_DIR

FIXTURE = Path(__file__).resolve().parents[1] / "roughness" / "fixtures" / "feature_parity.json"


@pytest.fixture(scope="module")
def measured() -> dict:
    return measure()


def test_simulator_is_deterministic_and_labelled() -> None:
    a = S.make_dataset(seed=5, walks=2, duration_s=40)
    b = S.make_dataset(seed=5, walks=2, duration_s=40)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"])
    assert a["x"].shape[1:] == (S.WINDOW, 6)
    c = S.make_dataset(seed=6, walks=2, duration_s=40)
    assert not np.array_equal(a["x"], c["x"])


def test_simulated_phone_reads_one_g_whatever_its_orientation() -> None:
    rng = np.random.default_rng(1)
    for _ in range(6):
        walk = S.simulate_walk(S.draw_persona(rng), rng, 30)
        assert abs(float(np.linalg.norm(walk.acc.mean(axis=0))) - S.G) < 0.6


def test_bump_is_labelled_where_its_response_is_visible() -> None:
    rng = np.random.default_rng(2)
    persona = S.draw_persona(rng, mode="roll")
    surface = np.zeros(20 * S.SIM_FS, dtype=np.int8)
    # 3.95 s is the last 50 ms of the second window, so the jolt shows up in the third.
    walk = S.simulate_walk(persona, rng, 20, timeline=(surface, [3.95, 9.0]))
    _, labels = S.label_windows(walk)
    assert labels[1] == S.SMOOTH and labels[2] == S.BUMP and labels[4] == S.BUMP
    assert list(labels[5:]) == [S.SMOOTH] * len(labels[5:])


def test_features_do_not_depend_on_how_the_phone_is_held() -> None:
    data = S.make_dataset(seed=9, walks=1, duration_s=20)
    window = data["x"][3].astype(np.float64)
    rotation = S._random_rotation(np.random.default_rng(4))
    turned = np.concatenate([window[:, :3] @ rotation.T, window[:, 3:] @ rotation.T], axis=1)
    a = F.extract(window[:, :3], window[:, 3:])
    b = F.extract(turned[:, :3], turned[:, 3:])
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_feature_fixture_matches_python_and_onnx() -> None:
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["feature_names"] == F.ALL_FEATURE_NAMES
    windows = np.array(fixture["windows"], dtype=np.float32)
    base = F.extract_many(windows)
    full = F.with_context(base, np.zeros(len(base), dtype=int))
    assert np.allclose(full, np.array(fixture["features"]), rtol=2e-3, atol=2e-3)
    session = ort.InferenceSession(str(MODEL_DIR / "roughness.onnx"), providers=["CPUExecutionProvider"])
    _, probs = session.run(None, {session.get_inputs()[0].name: np.array(fixture["features"], dtype=np.float32)})
    assert np.allclose(probs, np.array(fixture["probabilities"]), atol=1e-4)


def test_relative_features_are_neutral_until_there_is_a_baseline() -> None:
    base = F.extract_many(S.make_dataset(seed=3, walks=1, duration_s=30)["x"])
    assert list(F.relative(base[0], [])) == [1.0, 1.0, 0.0]
    assert list(F.relative(base[2], list(base[:2]))) == [1.0, 1.0, 0.0]
    assert F.relative(base[5], list(base[:5]))[0] != 1.0


def test_committed_metrics_are_reproduced_by_the_committed_model(measured: dict) -> None:
    recorded = json.loads(METRICS.read_text())["shipped_model"]
    for name in ("simulated_heldout", "simulated_heldout_shifted"):
        assert abs(measured[name]["macro_f1"] - recorded[name]["macro_f1"]) < 0.01, name


def test_macro_f1_on_heldout_simulated_walks_has_not_regressed(measured: dict) -> None:
    assert measured["simulated_heldout"]["macro_f1"] >= 0.80
    assert measured["simulated_heldout_shifted"]["macro_f1"] >= 0.78


@pytest.mark.xfail(strict=True, reason="PRD acceptance test not met: macro F1 on held-out SIMULATED walks is below 0.85, and no real labelled walk exists")
def test_acceptance_macro_f1_at_least_85_percent_on_a_heldout_walk(measured: dict) -> None:
    """Strict xfail: if a better model clears the bar this fails loudly, and the claim must be updated."""
    assert measured["simulated_heldout"]["macro_f1"] >= TARGET


def test_retrain_from_an_app_export_runs_end_to_end(tmp_path: Path) -> None:
    """The export here is synthetic. This checks the pipeline, not the model."""
    data = S.make_dataset(seed=21, walks=3, duration_s=60)
    doc = {
        "format": "curbcut-windows-v1",
        "rate_hz": S.FS,
        "window": S.WINDOW,
        "windows": [
            {"label": S.CLASSES[int(y)], "session": f"s{int(w)}", "index": i, "acc": x[:, :3].tolist(), "gyro": x[:, 3:].tolist()}
            for i, (x, y, w) in enumerate(zip(data["x"], data["y"], data["walk"]))
        ],
    }
    export = tmp_path / "export.json"
    export.write_text(json.dumps(doc))
    result = retrain.main([str(export), "--out", str(tmp_path), "--no-sim"])
    assert result["sessions"] == 3 and "leave one session out" in result["cross_validation"]["protocol"]
    assert 0.0 <= result["cross_validation"]["macro_f1"] <= 1.0
    assert (tmp_path / "roughness.onnx").stat().st_size > 1000
