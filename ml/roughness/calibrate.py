"""Side-by-side feature percentiles: simulated walkers against real gait recordings.

    .venv/bin/python -m ml.roughness.calibrate

This is how the walking half of the simulator was tuned. Only participants 1
to 20 of the real set are read here; 21 to 30 stay untouched for testing.
Simulated trunk is compared with the real lower-back sensor, simulated hand
with the real wrist, and simulated pocket with the real thigh. There is no
real reference for a bag, or for anything on wheels.
"""

from __future__ import annotations

import numpy as np

from . import features as F
from . import real_luo, simulator as S

PAIRS = [("trunk", "trunk"), ("hand", "wrist"), ("pocket", "thigh")]


def simulated(placement: str, seed: int = 5, people: int = 14) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(people):
        persona = S.draw_persona(rng, S.TRAIN_RANGES, mode="walk")
        persona = S.draw_persona(rng, S.TRAIN_RANGES, mode="walk") if persona.placement != placement else persona
        tries = 0
        while persona.placement != placement and tries < 50:
            persona = S.draw_persona(rng, S.TRAIN_RANGES, mode="walk")
            tries += 1
        # Mild rough ground only, the kind the real set has.
        persona = S.Persona(**{**persona.__dict__, "rough_scale": 1.0})
        surface, _ = S.draw_timeline(rng, 90.0)
        walk = S.simulate_walk(persona, rng, 90.0, timeline=(surface, []))
        x, y = S.label_windows(walk)
        xs.append(x)
        ys.append(y)
    return F.extract_many(np.concatenate(xs)), np.concatenate(ys)


def main() -> None:
    real = real_luo.load()
    keep = np.isin(real["participant"], real_luo.TRAIN_PARTICIPANTS)
    real_x = F.extract_many(real["x"][keep])
    for sim_name, real_name in PAIRS:
        sx, sy = simulated(sim_name)
        sel = real["sensor"][keep] == real_luo.SENSORS.index(real_name)
        rx, ry = real_x[sel], real["y"][keep][sel]
        print(f"\n== simulated {sim_name} vs real {real_name}: 10th / 50th / 90th percentile, smooth then rough")
        print(f"{'feature':16s} {'sim smooth':>22s} {'real smooth':>22s} {'sim rough':>22s} {'real rough':>22s}")
        for i, name in enumerate(F.FEATURE_NAMES):
            cells = []
            for x, y, c in ((sx, sy, 0), (rx, ry, 0), (sx, sy, 1), (rx, ry, 1)):
                q = np.percentile(x[y == c, i], [10, 50, 90])
                cells.append("%6.2f %6.2f %6.2f" % tuple(q))
            print(f"{name:16s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
