#!/usr/bin/env python3
"""Single-frame Gemini pose retarget (rough -> clean sketch)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai

from anime_frame.gemini.config import DEFAULT_MODEL, MODELS, load_api_key
from anime_frame.gemini.generate_sketch import generate_frame
from anime_frame.gemini.prompts_sketch import STYLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="luffy.jpg")
    ap.add_argument("--pose", default="Dataset/Naked Girl Kicking/Roughs/NakedGirlKicking_Roughs_0002.png")
    ap.add_argument("--out", default="outputs/clean_sketch.png")
    ap.add_argument("--style", default="line", choices=list(STYLES))
    ap.add_argument("--pose-mode", default="rough", choices=["rough", "rough_canny", "canny"])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--aspect", default="16:9")
    args = ap.parse_args()

    client = genai.Client(api_key=load_api_key())
    print(f"[gemini] model={args.model} style={args.style} pose_mode={args.pose_mode}")
    print(f"[gemini] target={args.target}")
    print(f"[gemini] pose  ={args.pose}")
    img = generate_frame(
        client, MODELS[args.model], args.target, args.pose,
        style=args.style, aspect_ratio=args.aspect, pose_mode=args.pose_mode,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(args.out)
    print(f"[gemini] saved -> {args.out}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
