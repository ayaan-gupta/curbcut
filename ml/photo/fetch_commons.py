"""Fetch openly licensed real photos of curbs from Wikimedia Commons.

    .venv/bin/python -m ml.photo.fetch_commons            # crawl categories, write the manifest, download
    .venv/bin/python -m ml.photo.fetch_commons --download # only download what the committed manifest lists

Photos are saved as 640 px wide JPEG thumbnails under data/photos/commons
(git-ignored). The manifest (ml/photo/manifest.json, committed) records the
source page, author, licence and SHA-256 of every file, so the exact set can be
fetched again and credited. Only files whose licence is CC0, public domain, CC
BY or CC BY-SA are kept.

The category a photo was found in is recorded but never used as its label.
Labels are assigned by looking at each photo, in labels.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
PHOTO_DIR = ROOT / "data" / "photos" / "commons"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "curbcut-dataset-build/0.1 (https://github.com/ayaan-gupta/curbcut)"}
THUMB_WIDTH = 640

# (category, how many files to take at most, whether to descend one level into subcategories)
CATEGORIES: list[tuple[str, int, bool]] = [
    ("Curb cuts", 10, False),
    ("Curb cuts in the United States", 30, True),
    ("Curb cuts in New York City", 22, False),
    ("Curb cuts in Israel", 22, False),
    ("Curb cuts in Moscow", 20, False),
    ("Curb cuts in Russia", 18, False),
    ("Curb cuts in Izmail", 10, False),
    ("Curb cuts in Hannover", 4, False),
    ("Curb cuts in Australia", 6, False),
    ("Curb cuts in California", 4, False),
    ("Curb cuts in Kyiv", 4, False),
    ("Kerb ramps in Japan", 30, False),
    ("Kerb ramps in Ballina, New South Wales", 12, False),
    ("Kerb ramps with tyre marks, Brisbane", 10, False),
    ("Kerb ramps in Brisbane, Queensland", 30, True),
    ("Kerbs", 40, False),
    ("Kerbs in Germany", 30, False),
    ("Kerbs in the United Kingdom", 28, False),
    ("Kerbs in London", 20, False),
    ("Kerbs in Tel Aviv-Yafo", 12, False),
    ("Kerbs in Azerbaijan", 8, False),
    ("Kerbs in the Czech Republic", 8, False),
    ("Curbs in the United States", 20, False),
    ("Curbs in Georgia (U.S. state)", 25, False),
    ("Curbs in Florida", 20, False),
    ("Curbs in New Orleans", 25, False),
    ("Curbs in California", 12, False),
    ("Curbs in Virginia", 10, False),
    ("Curbs in Ewing Township, New Jersey", 20, False),
    ("Curbstones", 20, False),
    ("Curbstones in Germany", 25, False),
    ("Curbs by condition", 60, True),
    ("Damaged sidewalks", 16, False),
    ("Damaged sidewalks in Brazil", 14, False),
    ("Damaged sidewalks in the United States", 8, False),
    ("Damaged pavements", 30, True),
    ("Sidewalk parking", 10, False),
    ("Sidewalk parking in Taipei", 25, False),
    ("Sidewalk parking in Taichung", 9, False),
    ("Sidewalk parking in Hsinchu City", 6, False),
    ("Cars parked on a sidewalk", 3, False),
    ("Sidewalk obstructions", 4, False),
    ("Sidewalk obstructions in Brazil", 5, False),
    ("Sidewalk obstructions in Taiwan", 20, True),
    ("Blocked footpaths", 30, True),
    ("Illegal parking in Germany", 25, False),
    ("Illegal parking in Munich", 20, False),
    ("Illegal parking in Berlin", 9, False),
    ("Illegal parking in France", 20, False),
    ("Illegal parking in Paris", 16, False),
    ("Illegal parking in Villejuif", 30, False),
    ("Illegal parking in Valencia", 20, False),
    ("Illegal parking in the United Kingdom", 13, False),
    ("Illegal parking in New Zealand", 11, False),
    ("Illegal parking in Taiwan", 25, False),
    ("Illegal parking in Ukraine", 12, False),
    ("Illegal parking in Russia", 12, False),
]

OPEN_LICENSE = re.compile(r"^(cc0|cc[ -]by([ -]sa)?[ -]\d|public domain|pd\b|cc-zero|no restrictions)", re.IGNORECASE)


def api(session: requests.Session, **params: Any) -> dict[str, Any]:
    for attempt in range(5):
        r = session.get(API, params={"format": "json", "formatversion": 2, **params}, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {r.status_code}")


def members(session: requests.Session, category: str, kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cont: dict[str, Any] = {}
    while True:
        data = api(session, action="query", list="categorymembers", cmtitle=f"Category:{category}", cmtype=kind, cmlimit=500, **cont)
        out.extend(data.get("query", {}).get("categorymembers", []))
        if "continue" not in data or len(out) >= 1500:
            return out
        cont = data["continue"]


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def describe(session: requests.Session, pageids: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(pageids), 40):
        data = api(
            session,
            action="query",
            pageids="|".join(str(p) for p in pageids[i : i + 40]),
            prop="imageinfo",
            iiprop="url|size|mime|extmetadata",
            iiurlwidth=THUMB_WIDTH,
        )
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [None])[0]
            if not info or info.get("mime") not in {"image/jpeg", "image/png"}:
                continue
            meta = info.get("extmetadata", {})
            license_name = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not OPEN_LICENSE.match(license_name):
                continue
            out.append(
                {
                    "id": page["pageid"],
                    "title": page["title"],
                    "source_url": info.get("descriptionurl", ""),
                    "thumb_url": info.get("thumburl", info.get("url", "")),
                    "author": strip_html(meta.get("Artist", {}).get("value", ""))[:200] or "unknown",
                    "license": license_name,
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "width": info.get("width"),
                    "height": info.get("height"),
                }
            )
        time.sleep(0.3)
    return out


def crawl() -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update(UA)
    rng = random.Random(9)
    chosen: dict[int, dict[str, Any]] = {}
    for category, limit, descend in CATEGORIES:
        cats = [category]
        if descend:
            cats += [c["title"].removeprefix("Category:") for c in members(session, category, "subcat")]
        pool: list[tuple[int, str]] = []
        for cat in cats:
            files = [m for m in members(session, cat, "file") if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))]
            rng.shuffle(files)
            # From a subcategory take a few at most: they tend to be one photographer's series of one place.
            take = files if cat == category else files[:3]
            pool.extend((m["pageid"], cat) for m in take)
            time.sleep(0.2)
        rng.shuffle(pool)
        fresh = [(pid, cat) for pid, cat in pool if pid not in chosen][: limit * 2]
        described = describe(session, [pid for pid, _ in fresh])
        by_id = dict(fresh)
        for entry in described[:limit]:
            entry["found_in"] = by_id[entry["id"]]
            chosen[entry["id"]] = entry
        print(f"{category}: kept {min(limit, len(described))} of {len(pool)}")
    return sorted(chosen.values(), key=lambda e: e["id"])


def download(entries: list[dict[str, Any]]) -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(UA)
    missing = 0
    for i, entry in enumerate(entries):
        path = PHOTO_DIR / f"{entry['id']}.jpg"
        if not path.exists():
            for attempt in range(4):
                r = session.get(entry["thumb_url"], timeout=60)
                if r.status_code == 200 and r.content[:2] in (b"\xff\xd8", b"\x89P"):
                    path.write_bytes(r.content)
                    break
                time.sleep(3 * (attempt + 1))
            else:
                missing += 1
                continue
            time.sleep(0.25)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry.get("sha256") and entry["sha256"] != digest:
            print(f"warning: {entry['id']} differs from the manifest (Commons re-rendered the thumbnail)")
        entry.setdefault("sha256", digest)
        if i % 50 == 49:
            print(f"  {i + 1}/{len(entries)}")
    print(f"{len(entries) - missing} photos present, {missing} could not be fetched")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="skip the crawl and fetch the files in the committed manifest")
    args = parser.parse_args()
    if args.download:
        entries = json.loads(MANIFEST.read_text())["photos"]
    else:
        entries = crawl()
    download(entries)
    if not args.download:
        MANIFEST.write_text(
            json.dumps(
                {
                    "note": "Real photos from Wikimedia Commons. Each file keeps its own licence; see source_url for the full attribution. The photos themselves are not committed.",
                    "thumb_width": THUMB_WIDTH,
                    "photos": entries,
                },
                indent=1,
                ensure_ascii=False,
            )
        )
        print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(entries)} photos")


if __name__ == "__main__":
    main()
