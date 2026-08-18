"""Shared Gemini API client helpers."""

from __future__ import annotations

import io
import time

from google.genai import types
from PIL import Image

from anime_frame.gemini.config import OVERLOAD_BACKOFF, REQUEST_TIMEOUT_MS


def safety_settings():
    cats = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    out = []
    for c in cats:
        for thr in ("OFF", "BLOCK_NONE"):
            try:
                out.append(types.SafetySetting(category=c, threshold=thr))
                break
            except Exception:  # noqa: BLE001
                continue
    return out or None


def extract_image(resp) -> Image.Image:
    cand = (resp.candidates or [None])[0]
    fr = getattr(cand, "finish_reason", None)
    pf = getattr(resp, "prompt_feedback", None)
    if cand is None or cand.content is None:
        raise RuntimeError(f"Empty response (finish_reason={fr}, prompt_feedback={pf})")
    for part in cand.content.parts or []:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return Image.open(io.BytesIO(part.inline_data.data))
    txt = " ".join(p.text for p in (cand.content.parts or []) if getattr(p, "text", None))
    raise RuntimeError(
        f"Model returned no image (finish_reason={fr}, prompt_feedback={pf}). Text: {txt!r}"
    )


def is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    keys = (
        "connection", "closed", "reset", "timeout", "timed out", "deadline",
        "temporarily", "unavailable", "internal", "503", "502", "500", "429",
    )
    return any(k in s for k in keys)


def is_overload(exc: Exception) -> bool:
    s = str(exc).lower()
    return "503" in s or "unavailable" in s or "high demand" in s or "overloaded" in s


def build_configs(*, aspect_ratio: str, seed: int, temperature: float):
    safety = safety_settings()
    return [
        types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            response_modalities=["Text", "Image"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            safety_settings=safety,
            seed=seed,
            temperature=temperature,
        ),
        types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            response_modalities=["Text", "Image"],
            safety_settings=safety,
            seed=seed,
            temperature=temperature,
        ),
        types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
            safety_settings=safety,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
        types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
    ]


def try_model(client, model, contents, configs, max_retries, tag="gemini"):
    last = None
    for cfg in configs:
        for attempt in range(max(max_retries, len(OVERLOAD_BACKOFF) + 1)):
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=cfg)
                return extract_image(resp)
            except Exception as e:  # noqa: BLE001
                last = e
                s = str(e)
                if is_overload(e):
                    if attempt < len(OVERLOAD_BACKOFF):
                        wait = OVERLOAD_BACKOFF[attempt]
                        print(
                            f"[{tag}] {model.split('/')[-1]} overloaded (503) -- waiting "
                            f"{wait}s, retry {attempt + 2}/{len(OVERLOAD_BACKOFF) + 1}",
                            flush=True,
                        )
                        time.sleep(wait)
                        continue
                    raise
                is_rate = "429" in s or "RESOURCE_EXHAUSTED" in s.upper() or "quota" in s.lower()
                if is_transient(e) and attempt < max_retries - 1:
                    wait = 15 * (attempt + 1) if is_rate else 2 ** attempt
                    what = "RATE LIMIT (429)" if is_rate else f"{type(e).__name__}: {s[:120]}"
                    print(f"[{tag}] {what} -> retry in {wait}s")
                    time.sleep(wait)
                    continue
                break
    raise last


def run_with_fallback(client, model_name, contents, configs, max_retries, *, fallback, tag):
    from anime_frame.gemini.config import MODELS

    chain = [model_name]
    fb = MODELS["gemini-3.1-flash-image"]
    if fallback and fb != model_name:
        chain.append(fb)

    last = None
    for i, model in enumerate(chain):
        try:
            return try_model(client, model, contents, configs, max_retries, tag=tag)
        except Exception as e:  # noqa: BLE001
            last = e
            if is_overload(e) and i < len(chain) - 1:
                print(f"[{tag}] falling back to {chain[i + 1].split('/')[-1]}")
                continue
            raise
    raise last
