"""Milestone 3: the photo classifier, its data, and the rule that synthetic images never reach evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from ml.photo import render_synth
from ml.photo.dataset import CLASSES, LABELS, MANIFEST, PHOTO_DIR, SYNTH_DIR, assert_real, counts, deploy_preprocess, load_photos
from ml.photo.evaluate import FIXTURE, MODEL, SAMPLE, TARGET, evaluate

METRICS = json.loads((Path(__file__).resolve().parents[1] / "photo" / "metrics.json").read_text())
OPEN = ("CC0", "Public domain", "CC BY", "PD", "No restrictions")


def test_every_manifest_photo_has_a_source_an_author_and_an_open_licence() -> None:
    photos = json.loads(MANIFEST.read_text())["photos"]
    assert len(photos) == 884
    for p in photos:
        assert p["source_url"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert p["author"] and p["license"].startswith(OPEN) and len(p["sha256"]) == 64


def test_labels_are_declared_as_model_labels_and_cover_only_manifest_photos() -> None:
    doc = json.loads(LABELS.read_text())
    assert "not human ground truth" in doc["labelled_by"]
    ids = {str(p["id"]) for p in json.loads(MANIFEST.read_text())["photos"]}
    assert set(doc["labels"]) <= ids
    assert set(doc["labels"].values()) <= {*CLASSES, "exclude"}
    assert doc["counts"] == {k: list(doc["labels"].values()).count(k) for k in sorted(set(doc["labels"].values()))}


def test_the_split_keeps_each_photographers_series_on_one_side() -> None:
    photos = load_photos()
    sides: dict[str, set[str]] = {}
    for p in photos:
        sides.setdefault(p.group, set()).add(p.split)
    assert all(len(s) == 1 for s in sides.values())
    assert counts(photos) == METRICS["real_photo_counts"]
    assert sum(counts(photos)["test"].values()) == METRICS["variants"][METRICS["shipped_variant"]]["test"]["n"]


def test_synthetic_images_are_labelled_and_can_never_be_evaluated_on(tmp_path: Path) -> None:
    written = render_synth.make_set(1, seed=3, out=tmp_path)
    assert len(written) == 4 and all(path.name.startswith("synthetic_") for path, _ in written)
    again = render_synth.make_set(1, seed=3, out=tmp_path / "again")
    assert np.array_equal(np.asarray(Image.open(written[0][0])), np.asarray(Image.open(again[0][0])))
    for path in (written[0][0], SYNTH_DIR / "present" / "synthetic_present_0000.jpg", tmp_path / "anything.jpg"):
        with pytest.raises(AssertionError):
            assert_real(path)
    assert_real(PHOTO_DIR / "123.jpg")
    assert all(PHOTO_DIR in p.path.parents for p in load_photos())


def test_the_exported_model_takes_raw_pixels_and_returns_four_probabilities() -> None:
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 224, 224]
    probs = session.run(None, {"pixels": deploy_preprocess(Image.open(SAMPLE).convert("RGB"))[None]})[0][0]
    assert probs.shape == (4,) and abs(float(probs.sum()) - 1.0) < 1e-4
    assert np.allclose(probs, json.loads(FIXTURE.read_text())["probs"], atol=0.03)  # JPEG decoders differ by a level or two across platforms


def test_recorded_accuracy_is_reproduced_when_the_real_photos_are_on_disk() -> None:
    test = [p for p in load_photos() if p.split == "test"]
    if not all(p.path.exists() for p in test):
        pytest.skip("real photos are not committed: run `python -m ml.photo.fetch_commons --download` to re-measure")
    recorded = METRICS["variants"][METRICS["shipped_variant"]]["test"]
    assert abs(evaluate(test)["accuracy"] - recorded["accuracy"]) < 0.03


@pytest.mark.xfail(strict=True, reason="PRD acceptance test not met: real-photo accuracy is below 0.85 (see ml/photo/metrics.json)")
def test_acceptance_photo_accuracy_at_least_85_percent_on_real_photos() -> None:
    """Strict xfail on the recorded number: if a retrained model clears the bar, this fails until the claim is updated."""
    shipped = METRICS["variants"][METRICS["shipped_variant"]]["test"]
    assert shipped["n"] >= 50, "too few real test photos to say anything"
    assert shipped["accuracy"] >= TARGET
