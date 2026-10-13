"""Fetch the face detector used for on-device blurring and tidy it for the browser.

    .venv/bin/python scripts/prepare_face_model.py

Model: Ultra-Light-Fast-Generic-Face-Detector-1MB, RFB-320 variant, by
Linzaer (MIT licence), as published in the ONNX Model Zoo:
https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface

It finds face boxes. It has no notion of who anyone is: there are no
embeddings and nothing to compare a face against.

The published file lists every weight as a graph input, which makes ONNX
Runtime print a warning per weight and skip constant folding. This removes
those inputs and a few unused bookkeeping tensors. The weights are untouched.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import onnx

URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
UPSTREAM_SHA256 = "34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017"
OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "models" / "face-ultraface-320.onnx"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(raw).hexdigest()
    print(f"downloaded {len(raw)} bytes, sha256 {digest}")
    if digest != UPSTREAM_SHA256:
        raise SystemExit("the upstream file is not the one this project was built with; check it before using it")
    model = onnx.load_from_string(raw)
    weights = {init.name for init in model.graph.initializer}
    kept = [inp for inp in model.graph.input if inp.name not in weights]
    del model.graph.input[:]
    model.graph.input.extend(kept)
    # Batch-norm bookkeeping tensors that no node reads.
    used = {name for node in model.graph.node for name in node.input}
    live = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(live)
    onnx.checker.check_model(model)
    OUT.write_bytes(model.SerializeToString())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), inputs: {[i.name for i in model.graph.input]}")


if __name__ == "__main__":
    main()
