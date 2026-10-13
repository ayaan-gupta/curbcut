"""Fine-tune MobileNetV3-Small to four curb states and export it for the browser.

    .venv/bin/python -m ml.photo.train

Three variants are trained from the same ImageNet weights:

    real            real training photos only
    real+synthetic  real training photos plus procedural scenes (labelled synthetic)
    synthetic       procedural scenes only, to show how far they are from photos

The variant that does best on the validation photos is shipped. The test
photos are scored once per variant, after that choice, and only ever contain
real photos. Every reported number comes from the exported ONNX file run with
the same preprocessing the phone uses.

Writes web/public/models/curb-photo.onnx (normalisation and softmax baked in:
input is RGB 0..255, NCHW, 224 x 224), curb-photo.json and ml/photo/metrics.json.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from . import render_synth
from .dataset import CLASSES, INPUT, ROOT, SYNTH_DIR, Photo, assert_real, counts, deploy_preprocess, load_image, load_photos

MODEL_DIR = ROOT / "web" / "public" / "models"
METRICS = Path(__file__).resolve().parent / "metrics.json"
RUNS = Path(__file__).resolve().parent / "runs"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 17
SYNTH_PER_CLASS = 300
EPOCHS_HEAD, EPOCHS_ALL = 4, 14
TARGET = 0.85


class Items(Dataset):
    def __init__(self, items: list[tuple[Path, int]], train: bool) -> None:
        self.items = items
        self.train = train
        self.augment = transforms.Compose(
            [
                transforms.RandomResizedCrop(INPUT, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(0.3, 0.3, 0.25, 0.03),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[i]
        img = load_image(path)
        if self.train:
            arr = np.asarray(self.augment(img), dtype=np.float32).transpose(2, 0, 1)
        else:
            arr = deploy_preprocess(img)
        return torch.from_numpy(np.ascontiguousarray(arr)), label


class Deployed(nn.Module):
    """The network with its preprocessing and softmax inside, so the phone only hands over pixels."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=1)


def build() -> Deployed:
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    return Deployed(net)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def predict(model: Deployed, items: list[tuple[Path, int]], dev: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in DataLoader(Items(items, train=False), batch_size=32):
            out.append(model(x.to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def fit(train: list[tuple[Path, int]], val: list[tuple[Path, int]], dev: torch.device, seed: int = SEED) -> tuple[Deployed, float]:
    """Trains one variant and returns the weights from its best validation epoch."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = build().to(dev)
    labels = np.array([lab for _, lab in train])
    class_count = np.bincount(labels, minlength=len(CLASSES)).clip(min=1)
    sampler = WeightedRandomSampler(weights=(1.0 / class_count)[labels], num_samples=max(len(train), 600), replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(Items(train, train=True), batch_size=32, sampler=sampler, num_workers=0)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_y = np.array([lab for _, lab in val])
    best_acc, best_state = -1.0, None
    for epoch in range(EPOCHS_HEAD + EPOCHS_ALL):
        head_only = epoch < EPOCHS_HEAD
        if epoch in (0, EPOCHS_HEAD):
            for p in model.net.features.parameters():
                p.requires_grad = not head_only
            params = [p for p in model.parameters() if p.requires_grad]
            optimiser = torch.optim.AdamW(params, lr=2e-3 if head_only else 3e-4, weight_decay=1e-4)
            steps = (EPOCHS_HEAD if head_only else EPOCHS_ALL) * len(loader)
            schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3 if head_only else 3e-4, total_steps=steps)
        model.train()
        if head_only:
            model.net.features.eval()  # keep batch-norm statistics while the backbone is frozen
        for x, y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model.logits(x.to(dev)), y.to(dev))
            loss.backward()
            optimiser.step()
            schedule.step()
        acc = float(np.mean(predict(model, val, dev).argmax(1) == val_y)) if len(val) else 0.0
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_acc


def export(model: Deployed, path: Path) -> None:
    model = model.cpu().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, INPUT, INPUT), str(path), input_names=["pixels"], output_names=["probs"], opset_version=17, dynamo=False)


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def evaluate_onnx(model_path: Path, photos: list[Photo]) -> dict[str, Any]:
    """Accuracy of an exported model on real photos, run the way the phone runs it."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    truth, pred = [], []
    for p in photos:
        assert_real(p.path)
        probs = session.run(None, {"pixels": deploy_preprocess(load_image(p.path))[None]})[0][0]
        truth.append(CLASSES.index(p.label))
        pred.append(int(np.argmax(probs)))
    y, yhat = np.array(truth), np.array(pred)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y, yhat):
        confusion[a, b] += 1
    correct = int(np.sum(y == yhat))
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = int(confusion[i, i])
        per_class[name] = {
            "support": int(confusion[i].sum()),
            "recall": round(tp / max(1, int(confusion[i].sum())), 4),
            "precision": round(tp / max(1, int(confusion[:, i].sum())), 4),
        }
    majority = int(np.bincount(y, minlength=len(CLASSES)).max()) if len(y) else 0
    return {
        "n": int(len(y)),
        "accuracy": round(correct / max(1, len(y)), 4),
        "accuracy_95ci_wilson": wilson(correct, len(y)),
        "balanced_accuracy": round(float(np.mean([per_class[c]["recall"] for c in CLASSES if per_class[c]["support"]])), 4),
        "majority_class_baseline": round(majority / max(1, len(y)), 4),
        "per_class": per_class,
        "confusion_rows_true_cols_pred": confusion.tolist(),
        "meets_target": bool(correct / max(1, len(y)) >= TARGET),
    }


def main() -> None:
    photos = [p for p in load_photos() if p.path.exists()]
    if not photos:
        raise SystemExit("no real photos on disk: run `python -m ml.photo.fetch_commons --download` first")
    by_split = {s: [p for p in photos if p.split == s] for s in ("train", "val", "test")}
    print("real photos by split:", json.dumps(counts(photos)))
    real_train = [(p.path, CLASSES.index(p.label)) for p in by_split["train"]]
    real_val = [(p.path, CLASSES.index(p.label)) for p in by_split["val"]]
    synth = [(path, CLASSES.index(label)) for path, label in render_synth.make_set(SYNTH_PER_CLASS, out=SYNTH_DIR)]
    print(f"{len(synth)} SYNTHETIC training images rendered")

    dev = device()
    variants = {"real": real_train, "real+synthetic": real_train + synth, "synthetic": synth}
    RUNS.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    for name, train in variants.items():
        model, val_acc = fit(train, real_val, dev)
        path = RUNS / f"{name.replace('+', '_')}.onnx"
        export(model, path)
        results[name] = {"train_images": len(train), "validation_accuracy": round(val_acc, 4), "test": evaluate_onnx(path, by_split["test"])}
        print(f"{name:16s} val {val_acc:.3f}  test {results[name]['test']['accuracy']:.3f}  (n={results[name]['test']['n']})")

    # Ship whichever of the two variants that saw real photos did better on validation. Ties go to the simpler one.
    shipped = max(("real", "real+synthetic"), key=lambda k: (results[k]["validation_accuracy"], k == "real"))
    final = RUNS / f"{shipped.replace('+', '_')}.onnx"
    (MODEL_DIR / "curb-photo.onnx").write_bytes(final.read_bytes())
    (MODEL_DIR / "curb-photo.json").write_text(
        json.dumps({"classes": CLASSES, "input": [1, 3, INPUT, INPUT], "range": "RGB 0..255", "variant": shipped, "base": "torchvision MobileNetV3-Small, ImageNet1K_V1 weights (BSD-3-Clause)"}, indent=1)
    )
    METRICS.write_text(
        json.dumps(
            {
                "note": "Test photos are real Wikimedia Commons photos only. Labels were assigned by the build agent (an AI model), not verified by a person. Synthetic images appear in training only.",
                "classes": CLASSES,
                "split_by": "author plus Commons category, so a photographer's series stays on one side",
                "real_photo_counts": counts(photos),
                "synthetic_training_images_per_class": SYNTH_PER_CLASS,
                "target_accuracy": TARGET,
                "shipped_variant": shipped,
                "shipped_selection": "best validation accuracy among the variants trained on real photos",
                "variants": results,
            },
            indent=1,
        )
    )
    shipped_test = results[shipped]["test"]
    print(f"shipped: {shipped}. Real-photo test accuracy {shipped_test['accuracy']:.3f} on {shipped_test['n']} photos, 95% CI {shipped_test['accuracy_95ci_wilson']}")
    print(f"model: {(MODEL_DIR / 'curb-photo.onnx').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
