"""Run logging — console + saved log files under the output directory."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_run_logger(out_dir: Path, run_name: str = "batch") -> logging.Logger:
    """
    Configure a logger that writes to:
      {out_dir}/logs/{run_name}_YYYYMMDD_HHMMSS.log  — archived run log
      {out_dir}/{run_name}.log                       — latest run (overwritten)

    Also mirrors messages to stdout.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = log_dir / f"{run_name}_{stamp}.log"
    latest = out_dir / f"{run_name}.log"

    logger = logging.getLogger(f"anime_frame.{run_name}.{stamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for path in (archived, latest):
        fh = logging.FileHandler(path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("run log archived -> %s", archived)
    logger.info("latest log       -> %s", latest)
    return logger
