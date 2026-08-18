#!/usr/bin/env python3
"""Remove background from line art or coloured cels."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_frame.paths import IMG_EXT
from anime_frame.postprocess import process_file


def main():
    ap = argparse.ArgumentParser(description="Remove background -> transparent PNG")
    ap.add_argument("--input", required=True)
    ap.add_argument("--kind", default="sketch", choices=["sketch", "color"])
    ap.add_argument("--out", help="output file (single input)")
    ap.add_argument("--out-dir", default="outputs/transparent")
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            raise SystemExit(f"no images in {src}")
        out_dir = Path(args.out_dir)
        for f in files:
            r = process_file(f, out_dir / f"{f.stem}.png", args.kind)
            print(f"[bg] {f.name} -> {out_dir / f.name}")
            print(f"     transparent={r['transparent_%']}%  ink={r['opaque_%']}%")
        print(f"[bg] done: {len(files)} file(s) -> {out_dir}")
    elif src.is_file():
        dst = Path(args.out) if args.out else Path("outputs/transparent") / f"{src.stem}.png"
        r = process_file(src, dst, args.kind)
        print(f"[bg] {src.name} -> {dst}")
        print(f"     transparent={r['transparent_%']}%  ink={r['opaque_%']}%")
    else:
        raise SystemExit(f"not found: {src}")


if __name__ == "__main__":
    main()
