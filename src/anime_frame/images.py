"""Image loading helpers."""

from PIL import Image


def load_rgb(path) -> Image.Image:
    """
    Load an image as RGB, flattening transparency onto white.

    Line-only transparent PNGs store strokes in the alpha channel; a plain
    .convert("RGB") would discard them and yield a solid black image.
    """
    im = path if isinstance(path, Image.Image) else Image.open(path)
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")
