"""Logging setup: different levels of messages to a file (and a clean console).

Design
------
*   One DEBUG   -> logs/debug.log     everything, including the noisy details
*   One INFO    -> logs/app.log       normal progress messages
*   One WARNING -> logs/app.log
*   One ERROR   -> logs/app.log
*   One CRITICAL-> logs/app.log

The console only shows INFO and above (colourised), so the terminal stays
readable while the log files keep the full detail. A rotating file handler keeps
the files from growing without bound.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# ------------------------------------------------------------------ colours --
class _Colour(logging.Formatter):
    """Short, colourised, one-line messages - never a traceback.

    `include_exception=False` is what keeps the console readable: the traceback
    is still written to the log files, the user only sees the message.
    """

    GREY = "\033[90m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"

    COLOURS = {
        logging.DEBUG: GREY,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, use_colour: bool = True, include_exception: bool = False):
        super().__init__()
        self.use_colour = use_colour
        self.include_exception = include_exception

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname.ljust(8)
        if self.use_colour:
            colour = self.COLOURS.get(record.levelno, "")
            head = f"{self.GREY}{stamp}{self.RESET} {colour}{level}{self.RESET}"
        else:
            head = f"{stamp} {level}"
        message = record.getMessage()
        if record.exc_info and self.include_exception:
            message = message + "\n" + self.formatException(record.exc_info)
        return f"{head} | {message}"


def _file_handler(path: str, level: int, max_bytes: int = 200_000,
                  backups: int = 3) -> logging.Handler:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


def setup_logging(log_dir: str = "logs", *, name: str = "app",
                  console_level: int = logging.INFO) -> logging.Logger:
    """Create the logger used by the whole program. Safe to call twice."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)          # the logger itself sees everything
    logger.propagate = False
    if logger.handlers:                     # already configured -> don't duplicate
        return logger

    # file 1: DEBUG only - the full trace of what happened
    logger.addHandler(_file_handler(os.path.join(log_dir, "debug.log"),
                                    logging.DEBUG))
    # file 2: INFO and above - what a human reads
    logger.addHandler(_file_handler(os.path.join(log_dir, "app.log"),
                                    logging.INFO))

    # console: INFO and above, colourised, message only (no traceback)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(_Colour(use_colour=sys.stdout.isatty(),
                                 include_exception=False))
    logger.addHandler(console)
    return logger


def log_level_table(logger: logging.Logger) -> None:
    """Log one message at every level so the routing is visible in the output."""
    logger.debug("DEBUG    - cache miss for key 'user:42', falling back to disk")
    logger.info("INFO     - 3 records written to output/records.csv")
    logger.warning("WARNING  - retrying upload, attempt 2 of 3")
    logger.error("ERROR    - record 7 could not be parsed as an integer")
    logger.critical("CRITICAL - database unreachable, shutting the worker down")
