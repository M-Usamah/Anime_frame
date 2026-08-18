#!/usr/bin/env python3
"""Colour a clean line sketch using a target character model sheet."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai

from anime_frame.gemini.config import DEFAULT_MODEL, MODELS, load_api_key
from anime_frame.gemini.generate_color import color_sketch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="outputs/colored.png")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--aspect", default="16:9")
    args = ap.parse_args()

    client = genai.Client(api_key=load_api_key())
    print(f"[color] model={args.model}")
    print(f"[color] sketch={args.sketch}")
    print(f"[color] target={args.target}")
    img = color_sketch(client, MODELS[args.model], args.sketch, args.target, aspect_ratio=args.aspect)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(args.out)
    print(f"[color] saved -> {args.out}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
