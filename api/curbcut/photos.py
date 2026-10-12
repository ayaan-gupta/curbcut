"""Uploaded photos. They arrive already blurred by the phone; the server only stores them for reviewers."""

from __future__ import annotations

import uuid
from pathlib import Path


class BadPhoto(ValueError):
    pass


def strip_jpeg_metadata(data: bytes) -> bytes:
    """Removes EXIF, XMP, IPTC and comment segments from a JPEG.

    The app re-encodes photos through a canvas, which already drops EXIF, so
    this is a second lock on the same door: a photo's embedded GPS position or
    phone serial number must never be stored.
    """
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise BadPhoto("not a JPEG")
    out = bytearray(b"\xff\xd8")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise BadPhoto("corrupt JPEG")
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is image data
            out += data[i:]
            return bytes(out)
        if marker == 0xFF:
            i += 1
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2 or i + 2 + length > len(data):
            raise BadPhoto("corrupt JPEG")
        drop = marker in (0xE1, 0xED, 0xFE)
        if not drop:
            out += data[i : i + 2 + length]
        i += 2 + length
    raise BadPhoto("JPEG has no image data")


def save_photo(data: bytes, directory: Path, max_bytes: int) -> str:
    if len(data) > max_bytes:
        raise BadPhoto(f"photo is larger than {max_bytes // 1000} kB")
    clean = strip_jpeg_metadata(data)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.jpg"
    (directory / name).write_bytes(clean)
    return name
