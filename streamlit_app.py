"""Streamlit Community Cloud entry point (share.streamlit.io)."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "streamlit_ui.py"), run_name="__main__")
