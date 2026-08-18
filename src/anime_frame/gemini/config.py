"""Gemini API configuration."""

import os

from anime_frame.paths import PROJECT_ROOT

MODELS = {
    "gemini-3-pro-image": "models/gemini-3-pro-image",
    "gemini-3.1-flash-image": "models/gemini-3.1-flash-image",
    "gemini-2.5-flash-image": "models/gemini-2.5-flash-image",
}
DEFAULT_MODEL = "gemini-3-pro-image"
REQUEST_TIMEOUT_MS = 120_000
OVERLOAD_BACKOFF = (8, 15, 30, 45, 60, 90)


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise SystemExit("GEMINI_API_KEY not found (env or .env)")
    return key
