"""Canny edge maps from rough animation frames."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _filter_construction_edges(edges: np.ndarray, gray: np.ndarray | None = None) -> np.ndarray:
    """Remove joint circles found on the grayscale rough (optional)."""
    out = edges.copy()
    if gray is None:
        return out

    h, w = gray.shape
    max_r = int(min(h, w) * 0.045)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
        param1=100, param2=35, minRadius=5, maxRadius=max(5, max_r),
    )
    if circles is None:
        return out

    max_area = np.pi * max_r * max_r
    for x, y, r in np.round(circles[0]).astype(int):
        mask = np.zeros_like(edges)
        cv2.circle(mask, (x, y), r, 255, -1)
        region = out[mask > 0]
        blob_area = int((region > 0).sum())
        if blob_area and blob_area < max_area * 0.9:
            cv2.circle(out, (x, y), r + 2, 0, -1)

    raw_count = int((edges > 0).sum())
    if int((out > 0).sum()) < max(500, raw_count * 0.05):
        return edges.copy()
    return out


def canny_from_rough(
    path,
    low: int = 80,
    high: int = 160,
    filter_construction: bool = False,
    dilate: int = 0,
    polarity: str = "black-on-white",
) -> Image.Image:
    """
    Canny edge map from a rough frame.

    polarity:
      black-on-white — for Gemini Option B
      white-on-black — OpenCV / ControlNet default
    """
    if polarity not in ("black-on-white", "white-on-black"):
        raise ValueError(f"polarity must be 'black-on-white' or 'white-on-black', got {polarity!r}")

    im = path if isinstance(path, Image.Image) else Image.open(path)
    gray = np.array(im.convert("L"))
    edges = cv2.Canny(gray, low, high)

    if filter_construction:
        edges = _filter_construction_edges(edges, gray=gray)

    if dilate > 0:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate)

    if polarity == "black-on-white":
        edges = 255 - edges

    return Image.fromarray(edges).convert("RGB")
