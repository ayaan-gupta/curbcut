"""Write the simulated sensor feed the app can replay when a device has no motion sensors.

    .venv/bin/python -m ml.roughness.make_demo_walk

SYNTHETIC. A simulated wheelchair with a frame-mounted phone rolls 150 m down
a real brick sidewalk in the pilot area (way 675018018, beside Cal Anderson
Park). The ground is simulated as rough near the seeded rough-surface report
and there are two simulated bumps. GPS fixes are the true position plus about
2 m of noise. The app labels this feed as simulated, and anything submitted
from it is stored with synthetic=true.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from . import simulator as S

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
from curbcut import routing  # noqa: E402
from curbcut.geo import haversine_m  # noqa: E402
from curbcut.graph import Graph  # noqa: E402

OUT = ROOT / "web" / "public" / "demo" / "walk.json"
START = (-122.32098, 47.61672)
END = (-122.32096, 47.61538)
ROUGH_AT = (-122.3209774, 47.615823)
BUMP_AT = (-122.3209875, 47.6165711)
SPEED = 1.25


def main() -> None:
    graph = Graph.load(ROOT / "data" / "pilot" / "capitol-hill.json")
    leg = routing.route(graph, {}, START, END, "walking").route
    assert leg is not None
    line = leg.geometry["coordinates"]
    cum = [0.0]
    for a, b in zip(line, line[1:]):
        cum.append(cum[-1] + haversine_m(a[0], a[1], b[0], b[1]))
    length = cum[-1]
    duration = math.floor(length / SPEED / 2) * 2.0

    def at(dist: float) -> tuple[float, float]:
        dist = min(max(dist, 0.0), length)
        i = max(1, next(k for k, c in enumerate(cum) if c >= dist))
        f = (dist - cum[i - 1]) / max(1e-9, cum[i] - cum[i - 1])
        return line[i - 1][0] + f * (line[i][0] - line[i - 1][0]), line[i - 1][1] + f * (line[i][1] - line[i - 1][1])

    def time_near(point: tuple[float, float]) -> float:
        ds = np.arange(0, length, 0.5)
        return float(ds[int(np.argmin([haversine_m(*at(d), *point) for d in ds]))] / SPEED)

    rng = np.random.default_rng(42)
    persona = S.draw_persona(rng, S.TRAIN_RANGES, mode="roll")
    persona = S.Persona(**{**persona.__dict__, "placement": "mount", "speed": SPEED, "push_amp": 0.0, "rough_scale": 1.6, "noise_acc": 0.05})
    surface = np.zeros(int(duration * S.SIM_FS), dtype=np.int8)
    t_rough = time_near(ROUGH_AT)
    surface[int((t_rough - 9) * S.SIM_FS) : int((t_rough + 9) * S.SIM_FS)] = S.ROUGH
    bumps = [time_near(BUMP_AT), min(duration - 5, t_rough + 22)]
    walk = S.simulate_walk(persona, rng, duration, timeline=(surface, bumps))
    _, labels = S.label_windows(walk)

    fixes = []
    for t in np.arange(0, duration + 0.01, 1.0):
        lon, lat = at(t * SPEED)
        fixes.append({"t": round(float(t) * 1000), "lon": round(lon + rng.normal(0, 2.0) / 75_000, 7), "lat": round(lat + rng.normal(0, 2.0) / 111_320, 7), "accuracy": round(float(rng.uniform(3, 6)), 1)})
    samples = np.concatenate([walk.acc, walk.gyro], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "note": "SYNTHETIC. A simulated wheelchair roll along a real sidewalk. Not a recording.",
                "rate_hz": S.FS,
                "duration_s": duration,
                "truth": [S.CLASSES[int(v)] for v in labels],
                "fixes": fixes,
                "samples": [round(float(v), 3) for v in samples.reshape(-1)],
            },
            separators=(",", ":"),
        )
    )
    print(f"wrote {OUT.relative_to(ROOT)}: {duration:.0f} s, {length:.0f} m, {OUT.stat().st_size / 1024:.0f} KB, truth {dict(zip(*np.unique(labels, return_counts=True)))}")


if __name__ == "__main__":
    main()
