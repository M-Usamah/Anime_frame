"""Anime pose-retargeting pipeline library."""

from anime_frame.canny import canny_from_rough
from anime_frame.gemini.generate_color import color_sketch
from anime_frame.gemini.generate_sketch import generate_frame
from anime_frame.postprocess import (
    desaturate_line_art,
    make_transparent,
    sketch_black_on_white,
)

__all__ = [
    "canny_from_rough",
    "color_sketch",
    "desaturate_line_art",
    "generate_frame",
    "make_transparent",
    "sketch_black_on_white",
]
