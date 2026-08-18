#!/usr/bin/env python3
"""Generate Canny edge maps from rough animation frames."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from anime_frame.canny import canny_from_rough
from anime_frame.paths import list_images


def main():
    ap = argparse.ArgumentParser(description="Generate Canny edge maps from rough frames")
    ap.add_argument("--input", type=Path, help="single rough image")
    ap.add_argument("--out", type=Path, help="output path (single input)")
    ap.add_argument("--roughs", type=Path, help="folder of rough frames")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/canny"))
    ap.add_argument("--low", type=int, default=80)
    ap.add_argument("--high", type=int, default=160)
    ap.add_argument("--filter-construction", action="store_true")
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--polarity", default="black-on-white",
                    choices=["black-on-white", "white-on-black"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    kwargs = dict(
        low=args.low, high=args.high,
        filter_construction=args.filter_construction,
        dilate=args.dilate, polarity=args.polarity,
    )

    if args.input:
        out = args.out or args.out_dir / f"{args.input.stem}_canny.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img = canny_from_rough(args.input, **kwargs)
        img.save(out)
        arr = np.array(img.convert("L"))
        edge_pct = (arr < 128).mean() * 100 if args.polarity == "black-on-white" else (arr > 128).mean() * 100
        print(f"[canny] {args.input.name} -> {out}  (edge {edge_pct:.2f}%)")
        return

    if not args.roughs:
        raise SystemExit("provide --input or --roughs")

    frames = list_images(args.roughs)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"no images in {args.roughs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rough in frames:
        out = args.out_dir / f"{rough.stem}_canny.png"
        canny_from_rough(rough, **kwargs).save(out)
        print(f"[canny] {rough.name} -> {out.name}")

    print(f"[canny] done -> {args.out_dir}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
