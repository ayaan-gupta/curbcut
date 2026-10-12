"""Roughness features for one 2 second window of phone motion.

The same arithmetic is implemented in web/src/lib/roughness/features.ts and a
shared fixture (ml/roughness/fixtures/feature_parity.json) keeps the two in
step. Change one and the parity tests on both sides fail until the other
follows.

Input: 100 samples at 50 Hz. Acceleration including gravity in the device
frame (m/s^2) and rotation rate (rad/s). The phone can be in any orientation:
gravity is taken as the window mean, and "vertical" is the component along it.
"""

from __future__ import annotations

import numpy as np

FS = 50
WINDOW = 100
EPS = 1e-9

FEATURE_NAMES = [
    "v_rms",
    "v_p2p",
    "v_maxabs",
    "v_crest",
    "v_kurtosis",
    "h_rms",
    "jerk_rms",
    "jerk_max",
    "band_0p5_3",
    "band_3_8",
    "band_8_15",
    "band_15_25",
    "hf_ratio",
    "dom_freq",
    "centroid",
    "gyro_rms",
    "gyro_max",
    "zcr",
    "burst_ratio",
    "top5_energy",
    "step_regularity",
]

# Spectral bands as inclusive DFT bin ranges. With N=100 and fs=50, bin k is 0.5*k Hz.
BANDS = [(1, 5), (6, 15), (16, 29), (30, 50)]
_HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / WINDOW)
_HANN_POWER = float(np.sum(_HANN**2))


def vertical_and_horizontal(acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits device-frame acceleration into the part along gravity and the rest."""
    g = acc.mean(axis=0)
    norm = float(np.linalg.norm(g))
    unit = g / (norm + EPS)
    along = acc @ unit
    vertical = along - norm
    horizontal = acc - np.outer(along, unit)
    return vertical, horizontal


def band_power(vertical: np.ndarray) -> np.ndarray:
    """Hann-windowed power per DFT bin, bins 0 to N/2."""
    spectrum = np.fft.rfft(vertical * _HANN)
    return (spectrum.real**2 + spectrum.imag**2) / _HANN_POWER


def extract(acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Returns the feature vector (float32, len(FEATURE_NAMES)) for one window."""
    if acc.shape != (WINDOW, 3) or gyro.shape != (WINDOW, 3):
        raise ValueError(f"expected two ({WINDOW}, 3) arrays, got {acc.shape} and {gyro.shape}")
    acc = acc.astype(np.float64)
    gyro = gyro.astype(np.float64)
    v, h = vertical_and_horizontal(acc)

    energy = float(np.sum(v**2))
    v_rms = float(np.sqrt(energy / WINDOW))
    v_maxabs = float(np.max(np.abs(v)))
    v_p2p = float(v.max() - v.min())
    v_crest = v_maxabs / (v_rms + EPS)
    v_kurtosis = float(np.mean(v**4) / (np.mean(v**2) ** 2 + EPS))
    h_rms = float(np.sqrt(np.mean(np.sum(h**2, axis=1))))

    jerk = np.diff(v) * FS
    jerk_rms = float(np.sqrt(np.mean(jerk**2)))
    jerk_max = float(np.max(np.abs(jerk)))

    power = band_power(v)
    bands = [float(np.sum(power[lo : hi + 1])) for lo, hi in BANDS]
    total = sum(bands)
    hf_ratio = (bands[2] + bands[3]) / (total + EPS)
    bins = np.arange(1, WINDOW // 2 + 1)
    freqs = bins * FS / WINDOW
    p = power[1:]
    dom_freq = float(freqs[int(np.argmax(p))])
    centroid = float(np.sum(freqs * p) / (np.sum(p) + EPS))

    w = gyro - gyro.mean(axis=0)
    w_mag2 = np.sum(w**2, axis=1)
    gyro_rms = float(np.sqrt(np.mean(w_mag2)))
    gyro_max = float(np.sqrt(np.max(w_mag2)))

    signs = v >= 0
    zcr = float(np.count_nonzero(signs[1:] != signs[:-1]) / WINDOW)

    # Ten consecutive 0.2 s blocks: how much the loudest one stands out from the typical one.
    block_rms = np.sqrt(np.mean(v.reshape(10, 10) ** 2, axis=1))
    burst_ratio = float(block_rms.max() / (np.sort(block_rms)[4] + 0.05))

    top5_energy = float(np.sum(np.sort(v**2)[-5:]) / (energy + EPS))

    # Gait periodicity: best normalised autocorrelation for lags of 0.3 to 0.8 s.
    best = -1.0
    for lag in range(15, 41):
        num = float(np.sum(v[:-lag] * v[lag:]))
        den = float(np.sqrt(np.sum(v[:-lag] ** 2) * np.sum(v[lag:] ** 2))) + EPS
        best = max(best, num / den)
    step_regularity = best

    return np.array(
        [
            v_rms,
            v_p2p,
            v_maxabs,
            v_crest,
            v_kurtosis,
            h_rms,
            jerk_rms,
            jerk_max,
            float(np.log10(bands[0] + 1e-6)),
            float(np.log10(bands[1] + 1e-6)),
            float(np.log10(bands[2] + 1e-6)),
            float(np.log10(bands[3] + 1e-6)),
            hf_ratio,
            dom_freq,
            centroid,
            gyro_rms,
            gyro_max,
            zcr,
            burst_ratio,
            top5_energy,
            step_regularity,
        ],
        dtype=np.float32,
    )


def extract_many(windows: np.ndarray) -> np.ndarray:
    """windows: (m, 100, 6) with acc in columns 0 to 2 and gyro in 3 to 5."""
    return np.stack([extract(w[:, :3], w[:, 3:]) for w in windows]) if len(windows) else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)


# --- Features relative to the walk so far -----------------------------------
#
# People differ far more from each other than smooth ground differs from rough
# ground under one person, so the classifier also sees each window relative to
# the same walk's recent past: the running median of up to BASELINE_WINDOWS
# earlier windows. With fewer than MIN_BASELINE earlier windows the relative
# features are neutral (1, 1, 0).

RELATIVE_NAMES = ["rel_v_rms", "rel_jerk_rms", "rel_hf"]
# The windows just before and just after say whether a jolt is a one-off (a bump)
# or part of a stretch (rough ground). The classifier therefore runs one window late.
CONTEXT_SOURCE = ["v_rms", "v_maxabs", "jerk_rms", "burst_ratio", "band_8_15", "band_15_25"]
CONTEXT_NAMES = [f"prev_{n}" for n in CONTEXT_SOURCE] + [f"next_{n}" for n in CONTEXT_SOURCE]
ALL_FEATURE_NAMES = FEATURE_NAMES + RELATIVE_NAMES + CONTEXT_NAMES
_CONTEXT_IDX = [FEATURE_NAMES.index(n) for n in CONTEXT_SOURCE]
BASELINE_WINDOWS = 20
MIN_BASELINE = 3
_V_RMS, _JERK_RMS, _B3, _B4 = (FEATURE_NAMES.index(n) for n in ("v_rms", "jerk_rms", "band_8_15", "band_15_25"))


def hf_log_power(base: np.ndarray) -> float:
    """log10 of the power above 8 Hz, rebuilt from the two logged band features."""
    return float(np.log10(10.0 ** float(base[_B3]) + 10.0 ** float(base[_B4])))


def relative(base: np.ndarray, history: list[np.ndarray]) -> np.ndarray:
    """The three relative features for one window given the base features of earlier windows, oldest first."""
    recent = history[-BASELINE_WINDOWS:]
    if len(recent) < MIN_BASELINE:
        return np.array([1.0, 1.0, 0.0], dtype=np.float32)
    med_v = float(np.median([h[_V_RMS] for h in recent]))
    med_j = float(np.median([h[_JERK_RMS] for h in recent]))
    med_hf = float(np.median([hf_log_power(h) for h in recent]))
    return np.array(
        [float(base[_V_RMS]) / (med_v + 1e-6), float(base[_JERK_RMS]) / (med_j + 1e-6), hf_log_power(base) - med_hf],
        dtype=np.float32,
    )


def assemble(base: np.ndarray, history: list[np.ndarray], prev: np.ndarray | None, nxt: np.ndarray | None) -> np.ndarray:
    """The full model input for one window.

    history: base features of earlier windows in this walk, oldest first.
    prev, nxt: base features of the neighbouring windows, or None at the ends
    of a walk, where the window's own values stand in.
    """
    before = base if prev is None else prev
    after = base if nxt is None else nxt
    return np.concatenate([base, relative(base, history), before[_CONTEXT_IDX], after[_CONTEXT_IDX]]).astype(np.float32)


def with_context(base: np.ndarray, walk_ids: np.ndarray) -> np.ndarray:
    """Model inputs for a whole dataset. Rows must be in time order within each walk."""
    out = np.zeros((len(base), len(ALL_FEATURE_NAMES)), dtype=np.float32)
    history: dict[int, list[np.ndarray]] = {}
    for i, walk in enumerate(walk_ids):
        past = history.setdefault(int(walk), [])
        prev = base[i - 1] if i > 0 and walk_ids[i - 1] == walk else None
        nxt = base[i + 1] if i + 1 < len(base) and walk_ids[i + 1] == walk else None
        out[i] = assemble(base[i], past, prev, nxt)
        past.append(base[i])
    return out
