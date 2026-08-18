"""Gemini cel-colouring generation."""

from anime_frame.gemini.client import build_configs, run_with_fallback
from anime_frame.gemini.prompts_color import CONSISTENCY_REF, PROMPT
from anime_frame.images import load_rgb


def color_sketch(
    client,
    model_name,
    sketch,
    target,
    aspect_ratio="16:9",
    max_retries=4,
    seed=0,
    temperature=0.25,
    anchor=None,
    fallback=True,
):
    contents = [PROMPT, load_rgb(sketch), load_rgb(target)]
    if anchor is not None:
        contents[0] = PROMPT + CONSISTENCY_REF
        contents.append(load_rgb(anchor))

    configs = build_configs(
        aspect_ratio=aspect_ratio, seed=seed, temperature=temperature,
    )
    return run_with_fallback(
        client, model_name, contents, configs, max_retries,
        fallback=fallback, tag="color",
    )
