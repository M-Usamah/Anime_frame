"""
Streamlit UI for the anime pose-retargeting pipeline.

Uses the same settings as scripts/gemini_batch.py:
  rough + Canny edge maps, line-art style, rolling previous-frame consistency.

Logs saved to: outputs/streamlit/ui.log and outputs/streamlit/logs/ui_*.log

Run:  streamlit run scripts/streamlit_ui.py
"""
import io
import logging
import re
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from google import genai
from PIL import Image

from anime_frame.gemini.config import MODELS, load_api_key
from anime_frame.gemini.generate_color import color_sketch
from anime_frame.gemini.generate_sketch import generate_frame
from anime_frame.images import load_rgb
from anime_frame.paths import IMG_EXT, PROJECT_ROOT
from anime_frame.postprocess import make_transparent, sketch_black_on_white
from anime_frame.run_log import setup_run_logger

MODEL = "gemini-3.1-flash-image"
SKETCH_STYLE = "line"
POSE_MODE = "rough_canny"
LOG_DIR = PROJECT_ROOT / "outputs" / "streamlit"

st.set_page_config(page_title="Anime Frame — Pose Retargeting", layout="wide")


def _get_logger(mode: str) -> logging.Logger:
    log = setup_run_logger(LOG_DIR, run_name="ui")
    log.info("--- streamlit run: %s ---", mode)
    log.info("model=%s  style=%s  pose_mode=%s", MODEL, SKETCH_STYLE, POSE_MODE)
    return log


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def _save_uploads(files) -> list[str]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    d = Path(tempfile.mkdtemp(prefix="upload_"))
    out = []
    for f in files:
        p = d / f.name
        p.write_bytes(f.getvalue())
        out.append(str(p))
    return sorted(out, key=_natural_key)


def _folder_images(path: str) -> list[str]:
    if not path or not path.strip():
        return []
    p = Path(path.strip().strip('"').strip("'")).expanduser()
    if not p.is_dir():
        return []
    files = [str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXT]
    return sorted(files, key=_natural_key)


def _frames(uploads, folder: str) -> list[str]:
    return _save_uploads(uploads) or _folder_images(folder)


def _one(upload, path: str) -> str | None:
    if upload:
        return _save_uploads(upload)[0]
    p = (path or "").strip().strip('"').strip("'")
    return p if p and Path(p).expanduser().is_file() else None


def _png_bytes(img: Image.Image, kind: str) -> bytes:
    buf = io.BytesIO()
    make_transparent(img, kind).save(buf, format="PNG")
    return buf.getvalue()


def _zip_bytes(images: list[Image.Image], prefix: str, kind: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, im in enumerate(images):
            zf.writestr(f"{prefix}_{i:04d}.png", _png_bytes(im, kind))
    return buf.getvalue()


def _save_images(images: list[Image.Image], prefix: str) -> list[str]:
    d = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    out = []
    for i, im in enumerate(images):
        p = d / f"{prefix}_{i:04d}.png"
        im.convert("RGB").save(p)
        out.append(str(p))
    return out


def _reason(err: str) -> str:
    if "IMAGE_SAFETY" in err:
        return "blocked by Gemini's image-safety filter (server-side, not disableable)"
    if "PROHIBITED_CONTENT" in err:
        return "blocked as prohibited content (server-side)"
    return err[:200]


def _run_sketch_batch(client, target_path: str, pose_paths: list[str], consistency: str, log: logging.Logger):
    """Match scripts/gemini_batch.py: rough + Canny, previous or anchor consistency."""
    results: list[Image.Image] = []
    blocked: list[tuple[str, str]] = []
    bar, status = st.progress(0.0), st.empty()

    previous: Image.Image | None = None
    fixed_anchor: Image.Image | None = None

    log.info("sketch batch: target=%s  frames=%d  consistency=%s", target_path, len(pose_paths), consistency)

    for i, p in enumerate(pose_paths):
        if consistency == "previous":
            ref = previous
            ref_label = "previous" if ref else "first frame"
        else:
            ref = fixed_anchor
            ref_label = "anchored" if ref else "setting anchor"

        status.write(f"Sketching frame {i + 1}/{len(pose_paths)} — `{Path(p).name}`  ({ref_label})")
        log.info("%d/%d %s (%s)", i + 1, len(pose_paths), Path(p).name, ref_label)
        try:
            img_raw = generate_frame(
                client, MODELS[MODEL], target_path, p,
                style=SKETCH_STYLE, pose_mode=POSE_MODE,
                anchor=ref, consistency_mode=consistency,
                fallback=False, seed=0, temperature=0.2,
            )
            img = sketch_black_on_white(img_raw, aggressive=True)
            results.append(img)
            log.info("OK %s", Path(p).name)
            if consistency == "previous":
                previous = img
            elif fixed_anchor is None:
                fixed_anchor = img
        except Exception as e:
            blocked.append((Path(p).name, _reason(str(e))))
            log.exception("SKIP %s", Path(p).name)
        bar.progress((i + 1) / len(pose_paths))

    status.empty()
    bar.empty()
    log.info("sketch done: ok=%d  blocked=%d", len(results), len(blocked))
    return results, blocked


def _run_color_batch(client, target_path: str, sketch_paths: list[str], log: logging.Logger):
    results: list = []
    blocked: list = []
    bar, status = st.progress(0.0), st.empty()
    anchor = None
    log.info("color batch: target=%s  frames=%d", target_path, len(sketch_paths))
    for i, p in enumerate(sketch_paths):
        status.write(f"Colouring frame {i + 1}/{len(sketch_paths)} — `{Path(p).name}`"
                     + ("" if anchor is None else "  (anchored)"))
        log.info("%d/%d %s", i + 1, len(sketch_paths), Path(p).name)
        try:
            out = color_sketch(client, MODELS[MODEL], p, target_path, anchor=anchor, fallback=False)
            results.append(out)
            log.info("OK %s", Path(p).name)
            if anchor is None:
                anchor = out
        except Exception as e:
            blocked.append((Path(p).name, _reason(str(e))))
            log.exception("SKIP %s", Path(p).name)
        bar.progress((i + 1) / len(sketch_paths))
    status.empty()
    bar.empty()
    log.info("color done: ok=%d  blocked=%d", len(results), len(blocked))
    return results, blocked


def run_sketch(target_path: str, pose_paths: list[str], consistency: str = "previous", *, mode: str = "sketch"):
    log = _get_logger(mode)
    client = genai.Client(api_key=load_api_key())
    return _run_sketch_batch(client, target_path, pose_paths, consistency, log)


def run_color(target_path: str, sketch_paths: list[str], *, mode: str = "color"):
    log = _get_logger(mode)
    client = genai.Client(api_key=load_api_key())
    return _run_color_batch(client, target_path, sketch_paths, log)


def show_blocked(blocked):
    if blocked:
        st.warning(f"⚠ {len(blocked)} frame(s) could not be generated:")
        for name, reason in blocked:
            st.write(f"- `{name}` — {reason}")


def show_results(images, prefix: str, key: str, kind: str):
    st.subheader(f"Results — {prefix} ({len(images)} frame(s))")
    cols = st.columns(min(3, len(images)))
    for i, im in enumerate(images):
        with cols[i % len(cols)]:
            st.image(im, use_container_width=True, caption=f"{prefix}_{i:04d}")

    label = "line-only transparent PNG" if kind == "sketch" else "transparent PNG"
    if len(images) == 1:
        st.download_button(
            f"⬇  Download result ({label})", data=_png_bytes(images[0], kind),
            file_name=f"{prefix}.png", mime="image/png",
            use_container_width=True, key=f"dl_{key}",
        )
    else:
        st.download_button(
            f"⬇  Download {len(images)} results ({label} zip)",
            data=_zip_bytes(images, prefix, kind),
            file_name=f"{prefix}_frames.zip", mime="application/zip",
            use_container_width=True, key=f"dl_{key}",
        )


def frame_inputs(key: str, frames_label: str):
    c1, c2 = st.columns(2)
    with c1:
        up_t = st.file_uploader(
            "① TARGET MODEL — the character / model sheet",
            type=["png", "jpg", "jpeg", "webp"], key=f"{key}_t",
        )
        path_t = st.text_input("…or target image path", key=f"{key}_tp",
                               placeholder=str(PROJECT_ROOT / "rias gremory.jpg"))
    with c2:
        up_f = st.file_uploader(
            f"② {frames_label} — select one or MANY frames",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True, key=f"{key}_f",
        )
        path_f = st.text_input("…or FOLDER path (loads every image in it)", key=f"{key}_fp",
                               placeholder="Dataset/Drive/Getting Out 1/Roughs")

    target = _one(up_t, path_t)
    frames = _frames(up_f, path_f)

    p1, p2 = st.columns([1, 2])
    with p1:
        st.caption("① Target model")
        if target:
            st.image(load_rgb(target), use_container_width=True)
        else:
            st.info("No target yet.")
    with p2:
        st.caption(f"② {frames_label} — {len(frames)} frame(s)")
        if frames:
            show = frames[:8]
            cols = st.columns(min(4, len(show)))
            for i, f in enumerate(show):
                with cols[i % len(cols)]:
                    st.image(load_rgb(f), use_container_width=True, caption=Path(f).name)
            if len(frames) > len(show):
                st.caption(f"…and {len(frames) - len(show)} more")
        else:
            st.info("No frames yet — upload several, or paste a folder path.")
    return target, frames


with st.sidebar:
    st.header("Pipeline settings")
    consistency = st.selectbox(
        "Sequence consistency",
        options=["previous", "anchor"],
        index=0,
        help="**previous** — each frame references the last output (default, matches gemini_batch.py). "
             "**anchor** — all frames lock to frame 1's design.",
    )
    st.caption(
        f"Model: `{MODEL}`  ·  Style: `{SKETCH_STYLE}`  ·  Pose: `{POSE_MODE}` "
        "(rough + Canny)"
    )
    st.divider()
    st.caption(
        "CLI equivalent:\n\n"
        "`python scripts/gemini_batch.py --target model.jpg "
        "--consistency previous --limit 10`"
    )
    st.divider()
    st.caption("**Logs**")
    st.code(str(LOG_DIR / "ui.log"), language=None)
    latest = LOG_DIR / "ui.log"
    if latest.exists():
        tail = latest.read_text(encoding="utf-8").splitlines()[-12:]
        with st.expander("Latest log tail"):
            st.text("\n".join(tail))

st.title("Anime Frame — Pose Retargeting")
st.caption(
    "Target model + rough pose frames → clean **black line art**. "
    "Uses rough + Canny guides and rolling previous-frame consistency "
    "(same as `scripts/gemini_batch.py`)."
)

tab_sketch, tab_color, tab_combine = st.tabs(["Model Pose Sketch", "Model Pose Color", "Combine"])

with tab_sketch:
    st.markdown("**Target model + pose rough(s) → clean black-line sketch**")
    s_target, s_frames = frame_inputs("s", "POSE TARGET — rough sketch frame(s)")

    if st.button("Generate sketch", type="primary", use_container_width=True, key="s_go"):
        if not s_target or not s_frames:
            st.error("Need a target model and at least one pose frame.")
        else:
            with st.spinner(f"Generating {len(s_frames)} frame(s)…"):
                res, blk = run_sketch(s_target, s_frames, consistency=consistency, mode="sketch-tab")
                st.session_state["s_res"], st.session_state["s_blk"] = res, blk

    if st.session_state.get("s_res"):
        show_results(st.session_state["s_res"], "sketch", "s", "sketch")
    show_blocked(st.session_state.get("s_blk", []))

with tab_color:
    st.markdown("**Target model + clean sketch(es) → coloured**")
    c_target, c_frames = frame_inputs("c", "SKETCH — clean sketch frame(s)")

    if st.button("Colour sketch", type="primary", use_container_width=True, key="c_go"):
        if not c_target or not c_frames:
            st.error("Need a target model and at least one sketch frame.")
        else:
            with st.spinner(f"Colouring {len(c_frames)} frame(s)…"):
                res, blk = run_color(c_target, c_frames, mode="color-tab")
                st.session_state["c_res"], st.session_state["c_blk"] = res, blk

    if st.session_state.get("c_res"):
        show_results(st.session_state["c_res"], "color", "c", "color")
    show_blocked(st.session_state.get("c_blk", []))

with tab_combine:
    st.markdown("**Step 1** makes the sketch → review → **Continue** colours it.")
    k_target, k_frames = frame_inputs("k", "POSE TARGET — rough sketch frame(s)")

    if st.button("1 · Generate sketch", type="primary", use_container_width=True, key="k_go1"):
        if not k_target or not k_frames:
            st.error("Need a target model and at least one pose frame.")
        else:
            with st.spinner(f"Stage 1 — sketching {len(k_frames)} frame(s)…"):
                res, blk = run_sketch(k_target, k_frames, consistency=consistency, mode="combine-sketch")
                st.session_state["k_sketch"] = res
                st.session_state["k_sketch_blk"] = blk
                st.session_state["k_sketch_paths"] = _save_images(res, "sketch")
                st.session_state["k_target_path"] = k_target
                st.session_state.pop("k_color", None)
                st.session_state.pop("k_color_blk", None)

    if st.session_state.get("k_sketch") is not None:
        if st.session_state["k_sketch"]:
            show_results(st.session_state["k_sketch"], "sketch", "k1", "sketch")
        show_blocked(st.session_state.get("k_sketch_blk", []))

        if st.session_state.get("k_sketch_paths"):
            st.divider()
            if st.button("2 · Continue → colour", type="primary", use_container_width=True, key="k_go2"):
                with st.spinner("Stage 2 — colouring…"):
                    res, blk = run_color(
                        st.session_state["k_target_path"],
                        st.session_state["k_sketch_paths"],
                        mode="combine-color",
                    )
                    st.session_state["k_color"], st.session_state["k_color_blk"] = res, blk

    if st.session_state.get("k_color") is not None:
        if st.session_state["k_color"]:
            show_results(st.session_state["k_color"], "color", "k2", "color")
        show_blocked(st.session_state.get("k_color_blk", []))
