"""Assignment 9 - demonstrate everything in order.

    python main.py

1. logging system  : one message per level, routed to the right file
2. custom exception: translate + clean up + protect the user
3. thread safe file: 8 threads writing the same file with and without a lock
"""
from __future__ import annotations

import os
import time

from core.exceptions import (AppError, ProcessingError, StorageError, TempFile,
                             ValidationError, clean_call, clean_method)
from core.logsetup import setup_logging, log_level_table
from core.safe_file import ThreadSafeFile, demo_race_condition, run_workers

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "logs")
OUT_DIR = os.path.join(BASE, "output")

# One logger for the whole program. setup_logging() is idempotent, so calling it
# again in a later step returns this same configured logger.
LOGGER = setup_logging(LOG_DIR)


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


# =========================================================== 1. LOGGING =====
def part_1_logging() -> None:
    banner("1. LOGGING SYSTEM - one message at every level")
    logger = setup_logging(LOG_DIR)
    logger.info("logging system ready, level=%s", logger.level)
    log_level_table(logger)

    print("\n  Files created:")
    for name in ("debug.log", "app.log"):
        path = os.path.join(LOG_DIR, name)
        with open(path, "r", encoding="utf-8") as handle:
            lines = [x for x in handle.read().splitlines() if x.strip()]
        print(f"    {name:<12} {len(lines):>3} lines   "
              f"(keeps: {'everything' if name.startswith('debug') else 'INFO and above'})")

    print("\n  Tail of logs/debug.log:")
    with open(os.path.join(LOG_DIR, "debug.log"), "r", encoding="utf-8") as handle:
        for line in handle.read().splitlines()[-5:]:
            print("    " + line)


# ======================================================== 2. EXCEPTIONS =====
# A small service whose methods are wrapped by the cleaning decorator, so any
# failure becomes a clean AppError instead of a raw built-in exception.
class RecordStore:
    def __init__(self, path: str, logger=LOGGER):
        self.path = path
        self.logger = logger
        self.records: list[dict] = []

    # The decorator translates + logs every exception this method raises.
    @clean_method(LOGGER, during="saving record")
    def save(self, name: str, score: int) -> dict:
        """Validate, then append. Partial writes are rolled back by TempFile."""
        if not isinstance(score, int):
            raise ValueError(f"score for {name!r} is {score!r}, not an int")
        if not 0 <= score <= 10:
            raise ValueError(f"score {score} for {name!r} is outside 0-10")
        if os.path.exists(self.path):
            os.remove(self.path)          # start clean so the demo is repeatable
        with TempFile(".record") as tmp:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(f"{name},{score}\n")
            os.replace(tmp, self.path)    # committed only on success
        record = {"name": name, "score": score}
        self.records.append(record)
        self.logger.info("stored record %s (score=%s)", name, score)
        return record

    @clean_method(LOGGER, during="loading records")
    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} does not exist yet")
        self.logger.debug("loading records from %s", self.path)
        return self.records

    @clean_method(LOGGER, during="crashing worker")
    def crash(self, worker: int) -> None:
        """A real bug: a ZeroDivisionError from deep inside the call stack."""
        self.logger.debug("worker %s dividing by its retry count", worker)
        return 10 / (worker - worker)      # always ZeroDivisionError


def part_2_exceptions() -> None:
    banner("2. CUSTOM EXCEPTION CLASS - exception cleaning")
    logger = setup_logging(LOG_DIR)
    store = RecordStore(os.path.join(OUT_DIR, "records.csv"), logger)

    print("\n  2a. Custom exception hierarchy")
    for exc in (AppError, ValidationError, StorageError, ProcessingError):
        print(f"    {exc.__name__:<18} code={exc.code:<18} {exc.status_hint}")

    print("\n  2b. Happy path - validated and stored")
    clean_call(store.save, "Attack on Titan", 9, during="saving record")
    clean_call(store.save, "Steins;Gate", 8, during="saving record")
    print(f"    stored {len(store.load())} records, file = "
          f"output/{os.path.basename(store.path)}")

    print("\n  2c. Exception cleaning in action")
    print("    input                              -> clean result")
    print("    " + "-" * 66)
    empty_store = RecordStore(os.path.join(OUT_DIR, "never_written.csv"), logger)
    cases = [
        ("score = 'nine'  (wrong type)", lambda: store.save("Cowboy Bebop", "nine")),
        ("score = 42      (out of range)", lambda: store.save("Cowboy Bebop", 42)),
        ("load() with no file on disk", lambda: empty_store.load()),
        ("crash() -> ZeroDivisionError", lambda: store.crash(3)),
    ]
    for label, action in cases:
        try:
            action()
            print(f"    {label:<36} -> no error (unexpected)")
        except AppError as err:
            cause = str(err.cause).replace(BASE + os.sep, "") if err.cause else ""
            print(f"    {label:<36} -> {err.code}")
            print(f"    {'':<36}    message: {err.message}")
            if cause:
                print(f"    {'':<36}    cause  : {type(err.cause).__name__}: {cause}")

    print("\n  2d. Exception chaining (__cause__) + cleanup")
    print("    traceback stored in logs/app.log, not shown to the user:")
    try:
        store.crash(3)
    except AppError as err:
        print(f"      caught {err.code}, chained from {type(err.cause).__name__}")
    leftovers = [f for f in os.listdir(OUT_DIR) if f.endswith(".tmp")]
    print(f"    temp files left behind after the failures: {len(leftovers)} "
          f"(TempFile cleaned up)")
    print(f"    records file still intact: {os.path.exists(store.path)} "
          f"with {len(store.load())} records")

    print("\n  2e. The user-friendly message vs the machine-readable one")
    err = ValidationError("score must be an integer", details={"name": "Cowboy Bebop"})
    print(f"      friendly  : {err}")
    print(f"      machine   : {err.as_dict()}")


# ================================================= 3. THREAD SAFE FILES ====
def part_3_threads() -> None:
    banner("3. THREAD SAFE FILE ACCESS")
    logger = setup_logging(LOG_DIR)

    safe_path = os.path.join(OUT_DIR, "concurrent.txt")
    racy_path = os.path.join(OUT_DIR, "unsafe.txt")
    for path in (safe_path, racy_path):
        if os.path.exists(path):
            os.remove(path)

    n_threads, n_lines = 8, 40
    print(f"\n  3a. {n_threads} threads x {n_lines} lines, WITHOUT a lock")
    expected, actual = demo_race_condition(racy_path, n_threads, n_lines)
    print(f"      expected {expected} lines, file has {actual} lines "
          f"-> {expected - actual} lost to the race")
    logger.warning("unsafe write lost %d of %d lines", expected - actual, expected)

    print("\n  3b. Same work WITH the lock + atomic replace")
    safe_file = ThreadSafeFile(safe_path)
    started = time.perf_counter()
    threads = run_workers(safe_file, n_threads, n_lines, delay=0.0005)
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started
    lines = safe_file.read_lines()
    print(f"      {len(threads)} threads finished in {elapsed:.3f}s")
    print(f"      expected {expected} lines, file has {len(lines)} lines "
          f"-> {'0 lost, no corruption' if len(lines) == expected else 'MISMATCH'}")
    print(f"      {safe_file.summary()}")
    logger.info("%d threads wrote %d lines with no corruption",
                n_threads, len(lines))

    print("\n  3c. First 6 lines as written (note the interleaved thread ids):")
    for line in lines[:6]:
        print("      " + line)

    print("\n  3d. Atomic write - readers never see a partial file")
    for n in (100, 250, 500):
        safe_file.write_atomic([f"snapshot line {i}" for i in range(n)])
        print(f"      after write_atomic({n:>3} lines) -> file reports "
              f"{len(safe_file.read_lines())} lines, no truncation")

    print("\n  3e. Batched append - one lock acquisition for many lines")
    safe_file.append_many([f"batch {i}" for i in range(10)])
    print(f"      {safe_file.summary()}")

    print("\n  Output files in output/:")
    for name in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, name))
        print(f"    {name:<22} {size:>6} bytes")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print("  ASSIGNMENT 9 - LOGGING, CUSTOM EXCEPTIONS, THREAD SAFE FILES")
    print("=" * 74)
    part_1_logging()
    part_2_exceptions()
    part_3_threads()
    print("\n" + "=" * 74)
    print("  DONE - see logs/debug.log, logs/app.log and output/")
    print("=" * 74)


if __name__ == "__main__":
    main()
