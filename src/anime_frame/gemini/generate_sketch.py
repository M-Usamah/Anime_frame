"""Gemini sketch / pose-retarget generation."""

from anime_frame.canny import canny_from_rough
from anime_frame.gemini.client import build_configs, run_with_fallback
from anime_frame.gemini.prompts_sketch import (
    CONSISTENCY_REF,
    CONSISTENCY_REF_PREVIOUS_ROUGH_CANNY,
    CONSISTENCY_REF_ROUGH_CANNY,
    ROUGH_CANNY_REF,
    STYLES,
)
from anime_frame.images import load_rgb


def generate_frame(
    client,
    model_name,
    target,
    pose,
    style="rough",
    aspect_ratio="16:9",
    max_retries=4,
    seed=0,
    temperature=0.25,
    anchor=None,
    fallback=True,
    pose_mode="rough",
    consistency_mode="anchor",
):
    """
    pose_mode:
      rough       — target + rough
      rough_canny — Option B: target + rough + Canny (+ optional anchor/previous)
      canny       — target + Canny only

    consistency_mode:
      anchor   — fixed frame-1 design lock
      previous — rolling previous-frame continuity
    """
    prompt = STYLES[style]
    if pose_mode == "rough_canny":
        prompt = prompt + ROUGH_CANNY_REF
        contents = [
            prompt,
            load_rgb(target),
            load_rgb(pose),
            canny_from_rough(pose, filter_construction=False),
        ]
        anchor_ref = (
            CONSISTENCY_REF_PREVIOUS_ROUGH_CANNY
            if consistency_mode == "previous"
            else CONSISTENCY_REF_ROUGH_CANNY
        )
    elif pose_mode == "canny":
        prompt = prompt + (
            "\nIMAGE 2 is a CANNY EDGE MAP (black silhouette edges on white). It defines "
            "ONLY pose, silhouette, and framing. Match it exactly.\n"
        )
        contents = [prompt, load_rgb(target), canny_from_rough(pose)]
        anchor_ref = (
            "\nIMAGE 3 is the ANCHOR: an approved frame from THIS SAME sequence. Use IMAGE 3 "
            "ONLY to keep SCALE and DESIGN identical — same character height, same hat and "
            "accessory sizes as IMAGE 3. Pose comes from IMAGE 2. DO NOT copy the anchor's pose."
        )
    else:
        contents = [prompt, load_rgb(target), load_rgb(pose)]
        anchor_ref = CONSISTENCY_REF

    if anchor is not None:
        contents[0] = prompt + anchor_ref
        contents.append(load_rgb(anchor))

    configs = build_configs(
        aspect_ratio=aspect_ratio, seed=seed, temperature=temperature,
    )
    return run_with_fallback(
        client, model_name, contents, configs, max_retries,
        fallback=fallback, tag="gemini",
    )
