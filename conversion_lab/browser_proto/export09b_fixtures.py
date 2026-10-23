"""Deterministic, non-evaluation fixtures for the export09b01 parity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "data/jee/training/next01/export09b01/fixtures"
SYNTHETIC_IMAGE = FIXTURE_DIR / "constant-acceleration.png"
QUESTION = "A particle starts at rest and moves with constant acceleration 2 m/s^2 for 3 s. What distance does it travel?"
INSTRUCTION = "Solve briefly, verify once, and finish with FINAL_ANSWER: followed by the numeric answer."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_synthetic_image() -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_IMAGE.exists():
        return SYNTHETIC_IMAGE
    image = Image.new("RGB", (1280, 240), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:
        font = ImageFont.load_default()
    lines = [
        "Synthetic export-parity fixture (not a JEE evaluation item)",
        "A particle starts at rest and moves with constant acceleration",
        "2 m/s^2 for 3 s. What distance does it travel?",
    ]
    y = 28
    for line in lines:
        draw.text((32, y), line, fill="black", font=font)
        y += 58
    temporary = SYNTHETIC_IMAGE.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(SYNTHETIC_IMAGE)
    return SYNTHETIC_IMAGE
