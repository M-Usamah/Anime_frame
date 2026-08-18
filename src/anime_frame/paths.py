"""Shared path helpers."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT],
        key=lambda p: natural_key(p.name),
    )
