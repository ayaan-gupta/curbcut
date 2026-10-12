"""A physically motivated simulator of phone motion while walking or rolling.

THIS IS SYNTHETIC DATA. Nothing here was recorded by a phone. It exists
because the project has no self-collected labelled walk yet; the app's
collection mode and retrain.py are how real windows replace it.

Two mechanisms are modelled:

- Walking. The trunk bounces once per step (a few centimetres, so a few m/s^2
  at normal cadence) and every heel strike adds a short damped oscillation.
  Rough ground makes step timing and impact size irregular and adds small
  impulses as the foot rocks on uneven stones. A bump is a step off or onto a
  lip: a brief dip, an outsized impact, and a disturbed step or two after.
- Rolling (wheelchair or stroller). There is no gait. The ground profile under
  the front caster and the rear wheel drives a damped spring (tyre and frame)
  and the frame acceleration is what the phone feels. Smooth concrete is
  low-amplitude noise with expansion joints, rough ground is pavers, gravel or
  broken slab, and a bump is a lip or root heave of 15 to 50 mm that the two
  axles hit one after the other.

On top of that sits where the phone is (belt, trouser pocket, hand, bag,
frame mount, lap), each with its own gain, low-pass, and wobble, a random
fixed orientation, sensor noise and bias. Everything is drawn from the ranges
in `Ranges`. Held-out walks use a different seed and, for the shifted set,
wider ranges than training ever saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

G = 9.80665
FS = 50
SIM_FS = 500
DECIM = SIM_FS // FS
WINDOW = 100
CLASSES = ["smooth", "rough", "bump"]
SMOOTH, ROUGH, BUMP = 0, 1, 2

WALK_PLACEMENTS = ["trunk", "pocket", "hand", "bag"]
ROLL_PLACEMENTS = ["mount", "lap"]


@dataclass(frozen=True)
class Ranges:
    """The distribution personas are drawn from. Tuples are uniform (low, high)."""

    name: str
    cadence_hz: tuple[float, float] = (1.55, 2.15)
    bounce_amp: tuple[float, float] = (1.7, 3.6)
    impact_gain: tuple[float, float] = (1.6, 3.6)
    step_jitter: tuple[float, float] = (0.02, 0.05)
    amp_var: tuple[float, float] = (0.08, 0.2)
    roll_speed: tuple[float, float] = (0.7, 1.5)
    wn_front_hz: tuple[float, float] = (9.0, 18.0)
    wn_rear_hz: tuple[float, float] = (5.0, 10.0)
    zeta: tuple[float, float] = (0.12, 0.35)
    noise_acc: tuple[float, float] = (0.02, 0.12)
    noise_gyro: tuple[float, float] = (0.003, 0.02)
    smooth_rms_mm: tuple[float, float] = (0.5, 1.6)
    rough_scale: tuple[float, float] = (0.8, 2.5)
    bump_mm: tuple[float, float] = (20.0, 50.0)
    walk_fraction: float = 0.5


TRAIN_RANGES = Ranges(name="train")
# Wider than training on every axis that matters: slower and faster people, noisier
# phones, softer and harder wheels, smaller bumps, and rough ground that is less rough.
SHIFTED_RANGES = Ranges(
    name="shifted",
    cadence_hz=(1.4, 2.3),
    bounce_amp=(1.4, 4.2),
    impact_gain=(1.3, 4.2),
    step_jitter=(0.02, 0.065),
    amp_var=(0.08, 0.26),
    roll_speed=(0.55, 1.8),
    wn_front_hz=(8.0, 21.0),
    wn_rear_hz=(4.0, 12.0),
    zeta=(0.1, 0.42),
    noise_acc=(0.02, 0.2),
    noise_gyro=(0.003, 0.035),
    smooth_rms_mm=(0.5, 2.0),
    rough_scale=(0.7, 2.8),
    bump_mm=(16.0, 55.0),
)


@dataclass(frozen=True)
class Persona:
    mode: str
    placement: str
    cadence_hz: float
    bounce_amp: float
    impact_gain: float
    impact_hz: float
    impact_tau: float
    step_jitter: float
    amp_var: float
    sway_amp: float
    speed: float
    wheelbase: float
    wn_front: float
    wn_rear: float
    zeta: float
    wheel_filter_m: float
    push_hz: float
    push_amp: float
    axle_mix: float
    gain: float
    rate_gain: float
    tissue_gain: float
    lowpass_hz: float
    wobble_deg: float
    wobble_hz: float
    noise_acc: float
    noise_gyro: float
    smooth_rms_mm: float
    rough_scale: float
    bump_mm: tuple[float, float]
    orientation: tuple[tuple[float, float, float], ...] = field(repr=False, default=((1, 0, 0), (0, 1, 0), (0, 0, 1)))


@dataclass
class Walk:
    acc: np.ndarray  # (n, 3) m/s^2, device frame, gravity included, 50 Hz
    gyro: np.ndarray  # (n, 3) rad/s, device frame, 50 Hz
    surface: np.ndarray  # (n,) 0 smooth, 1 rough
    bump_times: list[float]
    persona: Persona


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def draw_persona(rng: np.random.Generator, ranges: Ranges = TRAIN_RANGES, mode: str | None = None) -> Persona:
    def u(pair: tuple[float, float]) -> float:
        return float(rng.uniform(*pair))

    mode = mode or ("walk" if rng.random() < ranges.walk_fraction else "roll")
    placement = str(rng.choice(WALK_PLACEMENTS if mode == "walk" else ROLL_PLACEMENTS))
    # gain, rotation gain, soft-tissue gain, low-pass (Hz), wobble (degrees), wobble rate (Hz)
    gain, rate_gain, tissue_gain, lowpass, wobble_deg, wobble_hz = {
        "trunk": (rng.uniform(0.85, 1.2), 1.0, 1.0, rng.uniform(17, 23), rng.uniform(1.5, 4), rng.uniform(0.8, 2.0)),
        "pocket": (rng.uniform(0.95, 1.5), 2.3, rng.uniform(1.8, 3.0), rng.uniform(19, 24), rng.uniform(14, 28), 0.0),
        "hand": (rng.uniform(0.75, 1.1), 2.6, 0.7, rng.uniform(9, 15), rng.uniform(4, 11) if rng.random() < 0.6 else rng.uniform(2, 4), 0.0),
        "bag": (rng.uniform(0.6, 1.0), 1.2, 0.6, rng.uniform(5, 9), rng.uniform(4, 11), rng.uniform(1.0, 2.0)),
        "mount": (rng.uniform(1.0, 1.4), 1.0, 1.0, rng.uniform(18, 23), rng.uniform(0.3, 1.5), rng.uniform(0.5, 2.0)),
        "lap": (rng.uniform(0.45, 0.8), 1.0, 1.0, rng.uniform(6, 11), rng.uniform(1.5, 4), rng.uniform(0.5, 1.5)),
    }[placement]
    return Persona(
        mode=mode,
        placement=placement,
        cadence_hz=u(ranges.cadence_hz),
        bounce_amp=u(ranges.bounce_amp),
        impact_gain=u(ranges.impact_gain),
        impact_hz=float(rng.uniform(5, 11)),
        impact_tau=float(rng.uniform(0.03, 0.07)),
        step_jitter=u(ranges.step_jitter),
        amp_var=u(ranges.amp_var),
        sway_amp=float(rng.uniform(0.8, 1.8)),
        speed=u(ranges.roll_speed),
        wheelbase=float(rng.uniform(0.35, 0.55)),
        wn_front=2 * np.pi * u(ranges.wn_front_hz),
        wn_rear=2 * np.pi * u(ranges.wn_rear_hz),
        zeta=u(ranges.zeta),
        wheel_filter_m=float(rng.uniform(0.04, 0.09)),
        push_hz=float(rng.uniform(0.7, 1.2)),
        push_amp=float(rng.choice([0.0, rng.uniform(0.08, 0.2)])),
        axle_mix=float(rng.uniform(0.3, 0.7)),
        gain=float(gain),
        rate_gain=float(rate_gain),
        tissue_gain=float(tissue_gain),
        lowpass_hz=float(lowpass),
        wobble_deg=float(wobble_deg),
        wobble_hz=float(wobble_hz),
        noise_acc=u(ranges.noise_acc),
        noise_gyro=u(ranges.noise_gyro),
        smooth_rms_mm=u(ranges.smooth_rms_mm),
        rough_scale=u(ranges.rough_scale),
        bump_mm=ranges.bump_mm,
        orientation=tuple(tuple(float(v) for v in row) for row in _random_rotation(rng)),
    )


def draw_timeline(rng: np.random.Generator, duration_s: float, bump_every_s: float = 13.0) -> tuple[np.ndarray, list[float]]:
    """Per-sample surface at SIM_FS, and bump event times."""
    n = int(duration_s * SIM_FS)
    surface = np.zeros(n, dtype=np.int8)
    t = 0.0
    rough = rng.random() < 0.3
    while t < duration_s:
        length = float(rng.uniform(6, 20) if rough else rng.uniform(10, 34))
        if rough:
            surface[int(t * SIM_FS) : int(min(duration_s, t + length) * SIM_FS)] = ROUGH
        t += length
        rough = not rough
    bumps: list[float] = []
    t = float(rng.uniform(4, bump_every_s))
    while t < duration_s - 2:
        bumps.append(t)
        t += float(rng.uniform(4.5, 2 * bump_every_s))
    return surface, bumps


def _burst(n: int, start: int, freq: float, tau: float, amp: float) -> tuple[int, np.ndarray]:
    length = min(n - start, int(6 * tau * SIM_FS))
    if length <= 0:
        return start, np.zeros(0)
    tt = np.arange(length) / SIM_FS
    return start, amp * np.exp(-tt / tau) * np.sin(2 * np.pi * freq * tt)


def _walk_body(p: Persona, surface: np.ndarray, bumps: list[float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame linear acceleration (x forward, y left, z up), body rates, and stride phase.

    The shape and size of each term were set by comparing against real gait
    recordings (calibrate.py): heel strike dominates, the centre-of-mass bounce
    is the smaller part, and there is broadband content up to 20 Hz.
    """
    n = len(surface)
    acc = np.zeros((n, 3))
    rate = np.zeros((n, 3))
    stride_phase = np.zeros(n)
    envelope = np.ones(n)
    rs = p.rough_scale
    # A pocket feels its own leg land far harder than the other one.
    asym = rng.uniform(1.5, 2.2) if p.placement == "pocket" else rng.uniform(0.85, 1.15)
    hf_hz = rng.uniform(12, 20)
    hf_gain = rng.uniform(0.35, 0.8) * (1.8 if p.placement == "pocket" else 1.0)
    push_gain = rng.uniform(0.25, 0.55)

    # Step schedule.
    steps: list[list[float]] = []  # start, interval, amplitude, bump size, rough flag
    t = float(rng.uniform(0, 0.4))
    duration = n / SIM_FS
    while t < duration:
        rough = surface[min(n - 1, int(t * SIM_FS))] == ROUGH
        cadence = p.cadence_hz * (1 - rng.uniform(0.0, 0.05) * rs if rough else 1.0)
        jitter = p.step_jitter * (1 + 0.4 * rs if rough else 1.0)
        interval = max(0.28, (1 / cadence) * (1 + rng.normal(0, jitter)))
        amp = max(0.35, 1 + rng.normal(0, p.amp_var * (1 + 0.4 * rs if rough else 1.0)))
        if rough:
            amp *= 1 + 0.1 * rs
        steps.append([t, interval, amp, 0.0, 1.0 if rough else 0.0])
        t += interval
    starts = np.array([s[0] for s in steps])
    for bt in bumps:
        i = int(np.argmin(np.abs(starts - bt)))
        height = rng.uniform(*p.bump_mm) / 50.0  # 1.0 at a 50 mm lip
        steps[i][2] *= 1.5 + 2.0 * height * rng.uniform(0.7, 1.3)
        steps[i][3] = 1.0 + height
        if i + 1 < len(steps):
            steps[i + 1][2] *= rng.uniform(1.1, 1.5)
    # A stumble shifts every step after it.
    for i, s in enumerate(steps):
        if s[3] and i + 1 < len(steps):
            shift = steps[i][1] * rng.choice([-1, 1]) * rng.uniform(0.12, 0.3)
            for later in steps[i + 1 :]:
                later[0] += shift

    side = 1.0
    for start, interval, amp, bump, rough in steps:
        a = int(start * SIM_FS)
        b = min(n, int((start + interval) * SIM_FS))
        if a >= n or b <= a:
            side = -side
            continue
        amp_side = amp * (asym if side > 0 else 1 / asym)
        phase = 2 * np.pi * (np.arange(a, b) / SIM_FS - start) / interval
        acc[a:b, 2] += p.bounce_amp * amp_side * np.cos(phase)
        acc[a:b, 0] += 0.8 * p.bounce_amp * amp_side * np.sin(phase)
        acc[a:b, 1] += side * p.sway_amp * (1 + 0.15 * rs * rough) * np.sin(phase / 2)
        rate[a:b, 1] += 0.35 * np.sin(phase)
        rate[a:b, 0] += side * 0.3 * np.sin(phase / 2)
        rate[a:b, 2] += side * 0.3 * np.cos(phase / 2)
        stride_phase[a:b] = (phase / 2) + (0 if side > 0 else np.pi)
        envelope[a:b] = amp_side * (1 + 0.35 * rs * rough)
        # Heel strike: a low ring and a sharper one on top. Rough ground sharpens it.
        size = p.impact_gain * p.bounce_amp * amp_side
        s0, low = _burst(n, a, p.impact_hz * rng.uniform(0.85, 1.15), p.impact_tau, size)
        _, high = _burst(n, a, hf_hz * rng.uniform(0.85, 1.15), p.impact_tau * 0.6, size * hf_gain * (1 + 0.5 * rs * rough))
        wave = low.copy()
        wave[: len(high)] += high[: len(wave)]
        acc[s0 : s0 + len(wave), 2] += wave
        acc[s0 : s0 + len(wave), 0] -= 0.6 * wave
        acc[s0 : s0 + len(wave), 1] += 0.4 * side * wave
        rate[s0 : s0 + len(wave)] += np.outer(wave, [0.04 * side, 0.06, 0.03 * side])
        # Push off, a little past mid step.
        s1, push = _burst(n, a + int(rng.uniform(0.5, 0.65) * (b - a)), rng.uniform(3, 6), 0.07, size * push_gain)
        acc[s1 : s1 + len(push), 2] += push
        acc[s1 : s1 + len(push), 0] += 0.5 * push
        if bump:
            # The leading foot drops (or catches) before it lands: a short dip, then the big impact above.
            dip_len = int(rng.uniform(0.08, 0.16) * SIM_FS)
            d0 = max(0, a - dip_len)
            dip = -rng.uniform(1.5, 4.0) * bump * np.sin(np.pi * np.arange(a - d0) / max(1, a - d0))
            acc[d0:a, 2] += dip
            s2, spike = _burst(n, a, rng.uniform(3, 6), 0.08, rng.normal(0, 1.2) * bump)
            rate[s2 : s2 + len(spike), 1] += spike
        side = -side

    # Soft tissue and clothing: broadband motion that scales with how hard the person is stepping.
    sos = signal.butter(2, [3, 20], btype="bandpass", fs=SIM_FS, output="sos")
    tissue = signal.sosfilt(sos, rng.normal(size=(n, 3)), axis=0)
    tissue /= tissue.std() + 1e-12
    acc += tissue * (rng.uniform(0.25, 0.45) * p.tissue_gain * p.bounce_amp * envelope)[:, None]
    wobble = signal.sosfilt(sos, rng.normal(size=(n, 3)), axis=0)
    rate += wobble / (wobble.std() + 1e-12) * 0.2 * envelope[:, None]

    # Rough ground: the foot rocks on uneven stones between heel strike and toe off.
    rough_idx = np.nonzero(surface == ROUGH)[0]
    if len(rough_idx):
        lam = rng.uniform(3.0, 7.0)
        count = rng.poisson(lam * len(rough_idx) / SIM_FS)
        for start in rng.choice(rough_idx, size=count):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            amp = rng.normal(0, p.bounce_amp * rs * rng.uniform(0.15, 0.35))
            s0, wave = _burst(n, int(start), rng.uniform(8, 20), rng.uniform(0.02, 0.05), amp)
            acc[s0 : s0 + len(wave)] += np.outer(wave, direction)
            rate[s0 : s0 + len(wave)] += np.outer(wave, rng.normal(size=3)) * 0.12
    return acc, rate, stride_phase


def _shaped_noise(rng: np.random.Generator, n: int, dx: float, lo_wavelength: float, hi_wavelength: float, slope: float = 1.0) -> np.ndarray:
    """Unit-RMS noise whose amplitude spectrum falls as wavenumber^-slope inside a band."""
    spectrum = np.fft.rfft(rng.normal(size=n))
    k = np.fft.rfftfreq(n, d=dx)
    shape = np.zeros_like(k)
    band = (k >= 1 / hi_wavelength) & (k <= 1 / lo_wavelength)
    shape[band] = k[band] ** -slope
    out = np.fft.irfft(spectrum * shape, n=n)
    return out / (out.std() + 1e-12)


def _ground_profile(p: Persona, x: np.ndarray, surface: np.ndarray, bumps: list[float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Ground height (m) on a 2 mm grid covering the whole run."""
    dx = 0.002
    length = float(x[-1]) + p.wheelbase + 1.0
    n = int(length / dx)
    xg = np.arange(n) * dx
    z = _shaped_noise(rng, n, dx, 0.03, 3.0) * p.smooth_rms_mm / 1000.0

    # Expansion joints on most smooth sidewalks: a shallow groove every slab.
    if rng.random() < 0.7:
        slab = rng.uniform(1.2, 2.0)
        depth = rng.uniform(0.0015, 0.004)
        width = rng.uniform(0.008, 0.018)
        for joint in np.arange(rng.uniform(0, slab), length, slab):
            a, b = int(joint / dx), int((joint + width) / dx)
            z[a:b] -= depth

    # Rough stretches, each with its own texture.
    rough_t = np.nonzero(np.diff(np.concatenate([[0], (surface == ROUGH).astype(np.int8), [0]])))[0]
    for a_t, b_t in zip(rough_t[0::2], rough_t[1::2]):
        xa, xb = float(x[min(len(x) - 1, a_t)]) + p.wheelbase, float(x[min(len(x) - 1, b_t - 1)]) + p.wheelbase
        a, b = int(xa / dx), min(n, int(xb / dx))
        if b - a < 50:
            continue
        m = b - a
        kind = rng.choice(["cobble", "broken", "gravel"])
        scale = p.rough_scale
        if kind == "cobble":
            period = rng.uniform(0.08, 0.2)
            height = rng.uniform(0.003, 0.007) * scale
            tex = height * np.abs(np.sin(np.pi * np.arange(m) * dx / period)) + _shaped_noise(rng, m, dx, 0.02, 1.0) * rng.uniform(0.001, 0.002) * scale
        elif kind == "broken":
            tex = _shaped_noise(rng, m, dx, 0.03, 2.0) * rng.uniform(0.0025, 0.005) * scale
            for edge in np.nonzero(rng.random(m) < rng.uniform(0.6, 2.0) * dx)[0]:
                tex[edge:] += rng.choice([-1, 1]) * rng.uniform(0.003, 0.007) * scale
            tex -= np.linspace(tex[0], tex[-1], m)
        else:
            tex = _shaped_noise(rng, m, dx, 0.015, 0.25, slope=0.5) * rng.uniform(0.002, 0.0045) * scale
        z[a:b] += tex

    for bt in bumps:
        xe = float(x[min(len(x) - 1, int(bt * SIM_FS))]) + p.wheelbase
        e = int(xe / dx)
        h = rng.uniform(*p.bump_mm) / 1000.0
        if rng.random() < 0.6:
            z[e:] += rng.choice([-1, 1]) * h  # a lip up or a drop down
        else:
            half = int(rng.uniform(0.05, 0.15) / dx)
            ridge = h * (1 - np.abs(np.linspace(-1, 1, 2 * half)))
            z[e - half : e + half] += ridge[: len(z[e - half : e + half])]
    return xg, z


def _suspension(z: np.ndarray, wn: float, zeta: float) -> np.ndarray:
    """Acceleration of a mass on a damped spring whose base follows z(t)."""
    b, a = signal.bilinear([2 * zeta * wn, wn**2], [1, 2 * zeta * wn, wn**2], fs=SIM_FS)
    u = signal.lfilter(b, a, z - z[0]) + z[0]
    du = np.gradient(u) * SIM_FS
    dz = np.gradient(z) * SIM_FS
    return -2 * zeta * wn * (du - dz) - wn**2 * (u - z)


def _roll_body(p: Persona, surface: np.ndarray, bumps: list[float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(surface)
    t = np.arange(n) / SIM_FS
    slow = np.where(surface == ROUGH, rng.uniform(0.75, 0.95), 1.0)
    slow = signal.lfilter([0.004], [1, -0.996], slow, zi=[0.996 * slow[0]])[0]
    speed = p.speed * slow * (1 + p.push_amp * np.sin(2 * np.pi * p.push_hz * t + rng.uniform(0, 6.28)))
    x = np.cumsum(speed) / SIM_FS
    xg, z = _ground_profile(p, x, surface, bumps, rng)
    dx = xg[1] - xg[0]

    def wheel(z_in: np.ndarray, contact_m: float) -> np.ndarray:
        width = max(1, int(contact_m / dx))
        return np.convolve(z_in, np.ones(width) / width, mode="same")

    z_front = np.interp(x + p.wheelbase, xg, wheel(z, p.wheel_filter_m))
    z_rear = np.interp(x, xg, wheel(z, p.wheel_filter_m * 2.2))
    a_front = _suspension(z_front, p.wn_front, p.zeta)
    a_rear = _suspension(z_rear, p.wn_rear, min(0.6, p.zeta * 1.3))
    acc = np.zeros((n, 3))
    acc[:, 2] = p.axle_mix * a_front + (1 - p.axle_mix) * a_rear
    acc[:, 0] = np.gradient(speed) * SIM_FS - 0.25 * np.abs(acc[:, 2]) * np.sign(a_front)
    acc[:, 1] = 0.15 * acc[:, 2] * rng.uniform(-1, 1) + signal.lfilter([0.05], [1, -0.95], rng.normal(0, 0.15, n))
    rate = np.zeros((n, 3))
    rate[:, 1] = np.gradient((z_front - z_rear) / p.wheelbase) * SIM_FS
    rate[:, 0] = 0.3 * rate[:, 1] * rng.uniform(-1, 1)
    return acc, rate, np.zeros(n)


def simulate_walk(p: Persona, rng: np.random.Generator, duration_s: float = 180.0, timeline: tuple[np.ndarray, list[float]] | None = None) -> Walk:
    surface, bumps = timeline if timeline is not None else draw_timeline(rng, duration_s)
    n = len(surface)
    t = np.arange(n) / SIM_FS
    body, rate, stride_phase = (_walk_body if p.mode == "walk" else _roll_body)(p, surface, bumps, rng)

    # Where the phone is: gain and a low-pass for whatever sits between the body and the sensor.
    sos = signal.butter(2, min(p.lowpass_hz, 0.45 * SIM_FS), fs=SIM_FS, output="sos")
    body = signal.sosfiltfilt(sos, body, axis=0) * p.gain
    rate = signal.sosfiltfilt(sos, rate, axis=0) * p.rate_gain

    # Wobble about a mostly lateral axis. A pocket swings with the thigh at stride frequency.
    axis = np.array([0.15, 1.0, 0.1]) + rng.normal(0, 0.1, 3)
    axis /= np.linalg.norm(axis)
    amp = np.deg2rad(p.wobble_deg)
    if p.placement in ("pocket", "hand"):
        # A pocket swings with the thigh and a hand with the arm, both once per stride.
        theta = amp * np.sin(stride_phase)
        theta = signal.sosfiltfilt(signal.butter(2, 4, fs=SIM_FS, output="sos"), theta)
    else:
        theta = amp * np.sin(2 * np.pi * p.wobble_hz * t + rng.uniform(0, 6.28))
        theta += signal.sosfiltfilt(signal.butter(2, 0.5, fs=SIM_FS, output="sos"), rng.normal(0, amp * 4, n))
    dtheta = np.gradient(theta) * SIM_FS
    if p.placement in ("pocket", "hand"):
        # The phone rides on a lever: a quarter of a metre below the hip, or most of an arm below the shoulder.
        lever = 0.25 if p.placement == "pocket" else 0.3
        body[:, 0] += lever * np.gradient(dtheta) * SIM_FS
        body[:, 2] += lever * dtheta**2

    world = body + np.array([0.0, 0.0, G])
    # Rotate world into the wobbling frame (Rodrigues, angle -theta), then into the fixed device frame.
    c, s = np.cos(-theta)[:, None], np.sin(-theta)[:, None]
    wobbling = world * c + np.cross(axis, world) * s + np.outer(world @ axis, axis) * (1 - c)
    r0 = np.array(p.orientation)
    acc_dev = wobbling @ r0.T
    gyro_dev = (rate + np.outer(dtheta, axis)) @ r0.T

    acc50 = signal.decimate(acc_dev, DECIM, ftype="fir", axis=0, zero_phase=True)
    gyro50 = signal.decimate(gyro_dev, DECIM, ftype="fir", axis=0, zero_phase=True)
    m = len(acc50)
    acc50 = acc50 + rng.normal(0, p.noise_acc, (m, 3)) + rng.normal(0, 0.08, 3)
    gyro50 = gyro50 + rng.normal(0, p.noise_gyro, (m, 3)) + rng.normal(0, 0.01, 3)
    return Walk(acc=acc50.astype(np.float32), gyro=gyro50.astype(np.float32), surface=surface[::DECIM][:m].copy(), bump_times=list(bumps), persona=p)


BUMP_RESPONSE_S = 0.5
BUMP_MIN_OVERLAP_S = 0.15


def label_windows(walk: Walk, hop: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Cuts a walk into 2 s windows and labels each one.

    A bump is labelled where its response can be seen. The jolt and its ringing
    (and, on wheels, the second axle) last about half a second after the event,
    so a window is a bump if it overlaps that half second by at least 0.15 s.
    A bump in the last tenth of a second of a window therefore belongs to the
    next window, and one that straddles the edge belongs to both. Otherwise a
    window is rough if at least half of it is on rough ground, else smooth.
    Nothing is dropped: windows that straddle a surface change stay in and count.
    """
    n = len(walk.acc)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    bumps = np.array(walk.bump_times)
    for start in range(0, n - WINDOW + 1, hop):
        t0, t1 = start / FS, (start + WINDOW) / FS
        overlap = np.minimum(t1, bumps + BUMP_RESPONSE_S) - np.maximum(t0, bumps) if len(bumps) else np.zeros(0)
        if np.any(overlap >= BUMP_MIN_OVERLAP_S):
            label = BUMP
        elif walk.surface[start : start + WINDOW].mean() >= 0.5:
            label = ROUGH
        else:
            label = SMOOTH
        xs.append(np.concatenate([walk.acc[start : start + WINDOW], walk.gyro[start : start + WINDOW]], axis=1))
        ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


PLACEMENTS = WALK_PLACEMENTS + ROLL_PLACEMENTS


def make_dataset(seed: int, walks: int, ranges: Ranges = TRAIN_RANGES, duration_s: float = 180.0) -> dict[str, np.ndarray]:
    """Simulates whole walks. Returns windows in time order, labels, and per-window walk id, mode and placement."""
    rng = np.random.default_rng(seed)
    xs, ys, walk_ids, modes, placements = [], [], [], [], []
    for w in range(walks):
        persona = draw_persona(rng, ranges)
        walk = simulate_walk(persona, rng, duration_s)
        x, y = label_windows(walk)
        xs.append(x)
        ys.append(y)
        walk_ids.append(np.full(len(y), w))
        modes.append(np.full(len(y), 0 if persona.mode == "walk" else 1))
        placements.append(np.full(len(y), PLACEMENTS.index(persona.placement)))
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "walk": np.concatenate(walk_ids),
        "mode": np.concatenate(modes),
        "placement": np.concatenate(placements),
    }
