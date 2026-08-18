"""Line-art post-processing and transparent background removal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from anime_frame.paths import IMG_EXT


def ink_map(img: Image.Image, paper_cut: int = 60) -> np.ndarray:
    rgb = np.array(img.convert("RGB")).astype(np.float32)
    d = 255.0 - rgb.min(axis=2)
    d[d < paper_cut] = 0.0
    m = float(d.max())
    if m > 1.0:
        d *= 255.0 / m
    return np.clip(d, 0, 255).astype(np.uint8)


def hollow_fills(img: Image.Image, thr: int = 110, thickness: int = 4) -> Image.Image:
    g = np.array(img.convert("L"))
    dark = g < thr
    interior = ndimage.binary_erosion(dark, iterations=thickness)
    out = g.copy()
    out[interior] = 255
    return Image.fromarray(np.dstack([out, out, out]), "RGB")


def _normalize_stroke_width(dark: np.ndarray, target_median: float = 1.0) -> np.ndarray:
    if not dark.any():
        return dark
    struct = ndimage.generate_binary_structure(2, 2)
    dist = ndimage.distance_transform_edt(dark)
    med = float(np.median(dist[dark]))
    if med <= target_median + 0.25:
        return dark
    erode_n = max(1, int(round(med - target_median)))
    out = dark.copy()
    for _ in range(erode_n):
        out = ndimage.binary_erosion(out, structure=struct)
    return ndimage.binary_closing(out, structure=struct, iterations=1)


def desaturate_line_art(img: Image.Image) -> Image.Image:
    d = ink_map(img, paper_cut=50)
    g = 255 - d
    return Image.fromarray(np.dstack([g, g, g]), "RGB")


def clean_line_art(img: Image.Image, thin: bool = True) -> Image.Image:
    rgb = np.array(img.convert("RGB")).astype(np.float32)
    d = 255.0 - rgb.min(axis=2)
    d[d < 45] = 0.0
    m = float(d.max())
    if m > 1.0:
        d *= 255.0 / m

    dark = d > 28
    struct = ndimage.generate_binary_structure(2, 2)
    dark = ndimage.binary_closing(dark, structure=struct, iterations=1)

    labeled, n = ndimage.label(dark)
    if n:
        sizes = ndimage.sum(dark, labeled, range(1, n + 1))
        for i, sz in enumerate(sizes, 1):
            if sz < 8:
                dark[labeled == i] = False

    if thin:
        dark = _normalize_stroke_width(dark)

    g = np.where(dark, 0, 255).astype(np.uint8)
    img_rgb = Image.fromarray(np.dstack([g, g, g]), "RGB")
    return hollow_fills(img_rgb, thr=100, thickness=3)


def sketch_black_on_white(img: Image.Image, aggressive: bool = False) -> Image.Image:
    if aggressive:
        g = np.array(img.convert("L"))
        g = np.where(g <= 72, 0, 255).astype(np.uint8)
        dark = g < 128
        if dark.any():
            dist = ndimage.distance_transform_edt(dark)
            g[dist >= 2] = 255
        return Image.fromarray(np.dstack([g, g, g]), "RGB")
    g = 255 - ink_map(img, paper_cut=60)
    return hollow_fills(Image.fromarray(np.dstack([g, g, g]), "RGB"), thickness=4)


def sketch_transparent(img: Image.Image) -> Image.Image:
    d = ink_map(img)
    black = np.zeros(d.shape + (3,), dtype=np.uint8)
    return Image.fromarray(np.dstack([black, d]), "RGBA")


def color_transparent(img: Image.Image, thr: int = 238) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    near_white = (rgb >= thr).all(axis=2)
    lbl, n = ndimage.label(near_white)
    if n == 0:
        return img.convert("RGBA")
    border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border.discard(0)
    bg = np.isin(lbl, list(border)) if border else np.zeros_like(near_white)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def make_transparent(img: Image.Image, kind: str = "sketch") -> Image.Image:
    if kind not in ("sketch", "color"):
        raise ValueError(f"kind must be 'sketch' or 'color', got {kind!r}")
    return sketch_transparent(img) if kind == "sketch" else color_transparent(img)


def report(img: Image.Image, out: Image.Image) -> dict:
    a = np.array(out.convert("RGBA"))
    alpha, rgb = a[..., 3], a[..., :3]
    opaque = alpha > 0
    core = alpha > 200
    return {
        "transparent_%": round(float((alpha == 0).mean() * 100), 2),
        "opaque_%": round(float(opaque.mean() * 100), 2),
        "opaque_white_leftover_%": round(float(((rgb >= 238).all(-1) & opaque).mean() * 100), 3),
        "line_rgb": rgb[core].mean(0).round().astype(int).tolist() if core.any() else None,
    }


def process_file(src: Path, dst: Path, kind: str) -> dict:
    img = Image.open(src)
    out = make_transparent(img, kind)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return report(img, out)
