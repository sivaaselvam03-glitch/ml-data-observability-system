"""
=============================================================
MODULE: utils/logger.py
PURPOSE: Centralised logging setup for all modules.
=============================================================
"""

import os
import logging
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a logger that writes to both console and a date-stamped file.

    Parameters
    ----------
    name    : str – logger name (usually the module name)
    log_dir : str – directory for log files

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console handler (INFO level) ──────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (DEBUG level, new file each day) ─────
    log_file = os.path.join(
        log_dir,
        f"pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
