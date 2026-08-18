#!/usr/bin/env python3
"""
Batch pose retarget: Gemini + rough + Canny, with sequence consistency.

Logs are saved under --out-dir:
  batch.log              latest run
  logs/batch_*.log       archived per run

Usage:
    python scripts/gemini_batch.py --target "rias gremory.jpg" --prefix rias \\
        --consistency previous --limit 10 --out-dir outputs/gemini_option2_rias
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai
from PIL import Image

from anime_frame.canny import canny_from_rough
from anime_frame.gemini.config import MODELS, load_api_key
from anime_frame.gemini.generate_sketch import generate_frame
from anime_frame.paths import PROJECT_ROOT, list_images
from anime_frame.postprocess import desaturate_line_art, sketch_black_on_white
from anime_frame.run_log import setup_run_logger

DEFAULT_ROUGHS = PROJECT_ROOT / "Dataset/Drive/Getting Out 1/Roughs"
DEFAULT_TARGET = PROJECT_ROOT / "luffy.jpg"
DEFAULT_OUT = PROJECT_ROOT / "outputs/gemini_batch"
MODEL = "gemini-3.1-flash-image"

log: logging.Logger = logging.getLogger("anime_frame.batch")


def _clean_stale_outputs(out_dir: Path, prefix: str, keep_count: int) -> None:
    for p in sorted(out_dir.glob(f"{prefix}_*.png")):
        m = re.search(rf"{re.escape(prefix)}_(\d+)", p.name)
        if m and int(m.group(1)) >= keep_count:
            p.unlink()
            log.info("removed stale %s", p.name)


def main():
    global log

    ap = argparse.ArgumentParser(
        description="Gemini batch pose retarget (rough + Canny + consistency)",
    )
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                    help="target character model sheet")
    ap.add_argument("--roughs", type=Path, default=DEFAULT_ROUGHS,
                    help="folder of rough animation frames")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="output directory (creates raw/, canny/, logs/)")
    ap.add_argument("--prefix", default=None,
                    help="output filename prefix (default: target stem)")
    ap.add_argument("--limit", type=int, default=10,
                    help="max frames to process (0 = all)")
    ap.add_argument("--from-frame", type=int, default=0,
                    help="0-based index to start from")
    ap.add_argument("--frames", default=None,
                    help="comma-separated 1-based frame numbers (e.g. 3,9)")
    ap.add_argument("--consistency", default="previous", choices=["previous", "anchor"],
                    help="previous=rolling last frame; anchor=fixed frame-1 lock")
    ap.add_argument("--post-only", action="store_true",
                    help="re-run post-process on existing outputs (no API calls)")
    ap.add_argument("--from-raw", action="store_true",
                    help="with --post-only: read raw/ and write processed outputs")
    ap.add_argument("--rebuild-canny", action="store_true",
                    help="regenerate canny/ edge maps only (no API calls)")
    ap.add_argument("--style", default="line", choices=["rough", "line", "clean"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(args.out_dir, run_name="batch")

    prefix = args.prefix or args.target.stem
    all_frames = list_images(args.roughs)
    if args.limit:
        all_frames = all_frames[: args.limit]

    log.info("target=%s  roughs=%s  out=%s  prefix=%s", args.target, args.roughs, args.out_dir, prefix)
    log.info("model=%s  style=%s  consistency=%s  limit=%s", MODEL, args.style, args.consistency, args.limit)

    if args.frames:
        pick = {int(x.strip()) - 1 for x in args.frames.split(",") if x.strip()}
        work: list[tuple[int, Path]] = []
        for idx in sorted(pick):
            if idx < 0 or idx >= len(all_frames):
                log.error("frame %d out of range (1-%d)", idx + 1, len(all_frames))
                raise SystemExit(1)
            work.append((idx, all_frames[idx]))
    elif args.from_frame:
        work = [(args.from_frame + j, all_frames[args.from_frame + j])
                for j in range(len(all_frames) - args.from_frame)]
    else:
        work = list(enumerate(all_frames))

    if not work:
        log.error("no frames in %s", args.roughs)
        raise SystemExit(1)

    if not args.frames:
        _clean_stale_outputs(args.out_dir, prefix, keep_count=work[-1][0] + 1)

    if args.rebuild_canny:
        canny_dir = args.out_dir / "canny"
        canny_dir.mkdir(parents=True, exist_ok=True)
        for rough in all_frames:
            canny_path = canny_dir / f"{rough.stem}_canny.png"
            canny_from_rough(rough, filter_construction=False).save(canny_path)
            log.info("canny -> %s", canny_path.name)
        log.info("done rebuild-canny -> %s (%d frames)", canny_dir, len(all_frames))
        return

    if args.post_only:
        src_dir = args.out_dir / "raw" if args.from_raw else args.out_dir
        paths = sorted(src_dir.glob(f"{prefix}_*.png"))
        if not paths:
            log.error("no outputs in %s", src_dir)
            raise SystemExit(1)
        for p in paths:
            src = Image.open(p)
            img = sketch_black_on_white(src, aggressive=True) if args.from_raw else desaturate_line_art(src)
            dst = args.out_dir / p.name
            img.save(dst)
            log.info("post-processed %s -> %s", p.name, dst.name)
        log.info("done post-only -> %s (%d frames)", args.out_dir, len(paths))
        return

    client = genai.Client(api_key=load_api_key())
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    canny_dir = args.out_dir / "canny"
    canny_dir.mkdir(parents=True, exist_ok=True)

    fixed_anchor = None
    if args.consistency == "anchor":
        anchor_path = args.out_dir / f"{prefix}_0000.png"
        if anchor_path.exists():
            fixed_anchor = Image.open(anchor_path)
            log.info("loaded anchor from %s", anchor_path)

    blocked = []
    ok = 0
    total_out = len(all_frames)
    log.info("processing %d frame(s)%s",
             len(work),
             f"  pick={[i + 1 for i, _ in work]}" if args.frames else "")

    for i, rough in work:
        t0 = time.time()
        canny_path = canny_dir / f"{rough.stem}_canny.png"
        canny_from_rough(rough, filter_construction=False).save(canny_path)

        out_path = args.out_dir / f"{prefix}_{i:04d}.png"
        if args.consistency == "previous":
            ref_for_frame = None
            if i > 0:
                prev_path = args.out_dir / f"{prefix}_{i - 1:04d}.png"
                if prev_path.exists():
                    ref_for_frame = Image.open(prev_path)
            ref_label = "previous" if ref_for_frame else "first frame"
        else:
            ref_for_frame = fixed_anchor
            ref_label = "anchored" if ref_for_frame else "setting anchor"

        log.info("%d/%d %s (%s)", i + 1, total_out, rough.name, ref_label)
        try:
            img_raw = generate_frame(
                client, MODELS[MODEL], str(args.target), str(rough),
                style=args.style, pose_mode="rough_canny",
                anchor=ref_for_frame, consistency_mode=args.consistency,
                fallback=False, seed=0, temperature=0.2,
            )
            raw_dir.mkdir(parents=True, exist_ok=True)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            img_raw.save(raw_dir / f"{prefix}_{i:04d}.png")
            img = sketch_black_on_white(img_raw, aggressive=True)
            img.save(out_path)

            if args.consistency == "anchor" and fixed_anchor is None and i == 0:
                fixed_anchor = img

            elapsed = time.time() - t0
            ok += 1
            log.info("OK -> %s (%.0fs)", out_path, elapsed)
        except Exception:
            blocked.append(rough.name)
            log.exception("SKIP %s", rough.name)

    if blocked:
        log.error("finished with %d blocked / %d ok / %d total",
                  len(blocked), ok, len(work))
        for name in blocked:
            log.error("  blocked: %s", name)
        raise SystemExit(1)

    log.info("done -> %s (%d frames, all ok)", args.out_dir, len(work))


if __name__ == "__main__":
    main()
