"""Windows from the real Luo et al. (2020) gait recordings. See fetch_luo2020.py.

Real data, but not phone data: Xsens IMUs strapped to the lower back, the
wrist and the right thigh. Two surfaces map onto our classes (paved "flat
even" as smooth, "cobble stone" as rough). There are no bumps and no wheels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from .simulator import FS, ROUGH, SMOOTH, WINDOW

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "data" / "real" / "luo2020" / "trials.npz"
SENSORS = ["trunk", "wrist", "thigh"]
# Participants are split once, by person, so no one's gait is on both sides.
TEST_PARTICIPANTS = list(range(21, 31))
TRAIN_PARTICIPANTS = list(range(1, 21))


def available() -> bool:
    return PATH.exists()


def _label(trial: int) -> int:
    return SMOOTH if 4 <= trial <= 9 else ROUGH


def _is_walking(window: np.ndarray) -> bool:
    """Each trial starts and ends with the person standing still. Those windows are not walking on anything."""
    acc = window[:, :3]
    g = acc.mean(axis=0)
    vertical = acc @ (g / np.linalg.norm(g)) - np.linalg.norm(g)
    return float(vertical.std()) > 0.6


def load(sensors: list[str] | None = None) -> dict[str, np.ndarray]:
    """Returns windows (m, 100, 6), labels, participant, sensor index and trial for every usable 2 s window."""
    sensors = sensors or SENSORS
    data = np.load(PATH)
    rate = int(data["rate_hz"])
    xs, ys, people, sensor_ids, trials = [], [], [], [], []
    for key in data.files:
        if key == "rate_hz":
            continue
        p, t, s = key.split("_")
        if s not in sensors:
            continue
        raw = data[key]
        if len(raw) < 4 * rate:
            continue
        # NaN marks dropped packets too long to interpolate. Zero-fill for the filter, then mask.
        bad = np.isnan(raw).any(axis=1)
        clean = np.where(np.isnan(raw), 0.0, raw)
        down = signal.decimate(clean, rate // FS, ftype="fir", axis=0, zero_phase=True)
        bad_down = bad[:: rate // FS][: len(down)]
        for start in range(0, len(down) - WINDOW + 1, WINDOW):
            if bad_down[start : start + WINDOW].any():
                continue
            window = down[start : start + WINDOW].astype(np.float32)
            if not _is_walking(window):
                continue
            xs.append(window)
            ys.append(_label(int(t[1:])))
            people.append(int(p[1:]))
            sensor_ids.append(SENSORS.index(s))
            trials.append(int(t[1:]))
    return {
        "x": np.stack(xs),
        "y": np.array(ys, dtype=np.int64),
        "participant": np.array(people),
        "sensor": np.array(sensor_ids),
        "trial": np.array(trials),
    }


def spliced_walks(data: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Orders each person's trials at random and treats the result as one walk per person and sensor.

    Every trial was a separate short walk over one surface. The relative
    features need a walk that moves between surfaces, so the trials of one
    person are concatenated in a shuffled order. The windows are real; the
    order they follow each other in is not.
    """
    rng = np.random.default_rng(seed)
    order: list[int] = []
    walk_ids: list[int] = []
    for person in np.unique(data["participant"]):
        for sensor in np.unique(data["sensor"]):
            idx = np.nonzero((data["participant"] == person) & (data["sensor"] == sensor))[0]
            trials = np.unique(data["trial"][idx])
            rng.shuffle(trials)
            for trial in trials:
                order.extend(idx[data["trial"][idx] == trial].tolist())
                walk_ids.extend([int(person) * 10 + int(sensor)] * int(np.sum(data["trial"][idx] == trial)))
    out = {k: v[order] for k, v in data.items()}
    out["walk"] = np.array(walk_ids)
    return out
