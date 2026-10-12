"""Fetch a slice of a real, public-domain gait dataset for an honest sim-to-real check.

Luo, Coppola, Dixon, Li, Dennerlein and Hu (2020), "A database of human gait
performance on irregular and uneven surfaces collected by wearable sensors",
Scientific Data 7:219. Data: https://doi.org/10.6084/m9.figshare.c.4892463
(CC0). Thirty people walked outdoors wearing six Xsens IMUs at 100 Hz.

What this is and is not:
- real accelerometer and gyroscope data from people walking on real surfaces
- body-worn research IMUs, not phones
- only two of our three classes exist in it: "flat even" paved ground
  (trials 4 to 9, used as smooth) and "cobble stone" uneven stone brick
  (trials 10 to 15, used as rough). There is no bump class and nobody rolls.

The archive is a single 2 GB zip. Only about 5 percent of it is needed, so
this reads the zip directory over HTTP range requests and pulls the trunk,
wrist and right-thigh files for those twelve trials.

    .venv/bin/python ml/roughness/fetch_luo2020.py

Writes data/real/luo2020/trials.npz (git-ignored, about 25 MB).
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "real" / "luo2020" / "trials.npz"
URL = "https://ndownloader.figshare.com/files/21998373"
SENSORS = {"CC": "trunk", "95": "wrist", "93": "thigh"}
TRIALS = {**{t: "smooth" for t in range(4, 10)}, **{t: "rough" for t in range(10, 16)}}
RATE_HZ = 100


class HttpFile(io.RawIOBase):
    """A read-only seekable file over HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.session = requests.Session()
        first = self.session.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        first.raise_for_status()
        self.url = first.url
        self.size = int(first.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(4):
            r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
            if r.status_code in (200, 206):
                self.pos += len(r.content)
                return r.content
        raise OSError(f"range request failed with {r.status_code}")

    def readinto(self, buffer: bytearray) -> int:  # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def parse_trial(text: str) -> np.ndarray:
    """Returns an (n, 6) float32 array of acc xyz (m/s^2, with gravity) and gyro xyz (rad/s).

    Dropped packets are filled by linear interpolation when the gap is at most
    five frames. A longer gap is filled with NaN so windows over it are skipped.
    """
    lines = text.splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    cols = [header.index(c) for c in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")]
    counter_col = header.index("PacketCounter")
    counters: list[int] = []
    rows: list[list[float]] = []
    for line in lines[1:]:
        parts = line.split(",")
        try:
            row = [float(parts[c]) for c in cols]
            counters.append(int(parts[counter_col]))
            rows.append(row)
        except (ValueError, IndexError):
            continue
    if len(rows) < 10:
        return np.zeros((0, 6), dtype=np.float32)
    t = np.array(counters, dtype=np.int64)
    t = np.unwrap(t, period=65536) - t[0]
    values = np.array(rows, dtype=np.float64)
    full = np.arange(int(t[-1]) + 1)
    out = np.stack([np.interp(full, t, values[:, i]) for i in range(6)], axis=1)
    gaps = np.diff(t)
    for idx in np.nonzero(gaps > 5)[0]:
        out[int(t[idx]) + 1 : int(t[idx + 1])] = np.nan
    return out.astype(np.float32)


def main() -> None:
    remote = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))  # type: ignore[arg-type]
    wanted: list[tuple[str, zipfile.ZipInfo]] = []
    for info in remote.infolist():
        parts = info.filename.split("/")
        if len(parts) != 3 or not parts[2].endswith(".txt.csv"):
            continue
        participant, name = parts[1], parts[2]
        trial = int(name.split("-")[0])
        sensor = name.split(".")[0][-2:]
        if trial in TRIALS and sensor in SENSORS:
            wanted.append((f"p{int(participant):02d}_t{trial:02d}_{SENSORS[sensor]}", info))
    print(f"{len(wanted)} files, {sum(i.compress_size for _, i in wanted) / 1e6:.0f} MB compressed")

    def fetch(item: tuple[str, zipfile.ZipInfo]) -> tuple[str, np.ndarray]:
        key, info = item
        # One zip handle per call: ZipFile is not safe to share across threads.
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 18))  # type: ignore[arg-type]
        return key, parse_trial(z.read(info.filename).decode("utf8", "replace"))

    arrays: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (key, arr) in enumerate(pool.map(fetch, wanted)):
            arrays[key] = arr
            if i % 100 == 99:
                print(f"  {i + 1}/{len(wanted)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, rate_hz=RATE_HZ, **arrays)  # type: ignore[arg-type]
    seconds = sum(len(a) for a in arrays.values()) / RATE_HZ
    print(f"wrote {OUT.relative_to(ROOT)}: {len(arrays)} trials, {seconds / 60:.0f} minutes of signal")


if __name__ == "__main__":
    main()
