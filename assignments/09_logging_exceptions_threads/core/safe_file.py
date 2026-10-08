"""Thread safe file access.

Three techniques are combined, because each solves a different problem:

1. A LOCK          - stops two threads interleaving their writes (see
                      demo_race_condition for what happens without it).
2. ATOMIC WRITES   - data goes to a temp file first and is then renamed with
                      os.replace(), so a reader never sees a half-written file.
3. A RE-ENTRANT    - the lock is an RLock, so a method may safely call another
   LOCK              method of the same object.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Iterable


def _replace_with_retry(src: str, dst: str, attempts: int = 5,
                        delay: float = 0.05) -> None:
    """os.replace() with a short retry.

    os.replace() is atomic, but Windows (and cloud-sync folders such as OneDrive)
    can hold a transient lock on the destination, which surfaces as
    PermissionError. Retrying briefly is the standard fix.
    """
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)


class ThreadSafeFile:
    """Append-to / write-to a text file from any number of threads."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()          # re-entrant
        self.write_count = 0

    # ------------------------------------------------------------- helpers --
    def _with_lock(self, action: Callable[[], None]) -> None:
        with self._lock:                       # every access goes through here
            action()
            self.write_count += 1

    # ------------------------------------------------------------- reading --
    def read_all(self) -> str:
        with self._lock:
            if not os.path.exists(self.path):
                return ""
            with open(self.path, "r", encoding="utf-8") as handle:
                return handle.read()

    def read_lines(self) -> list[str]:
        text = self.read_all()
        return [line for line in text.splitlines() if line.strip()]

    # ------------------------------------------------------------- writing --
    def append(self, line: str) -> None:
        """Append one line atomically with respect to other threads."""
        def action() -> None:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line.rstrip("\n") + "\n")
                handle.flush()
        self._with_lock(action)

    def write_atomic(self, lines: Iterable[str]) -> None:
        """Replace the whole file in one step, never leaving it half written."""
        payload = "\n".join(str(x) for x in lines) + "\n"

        def action() -> None:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())      # force it to disk
            _replace_with_retry(tmp, self.path)
        self._with_lock(action)

    def append_many(self, lines: Iterable[str]) -> None:
        """Batched append - one lock acquisition for many lines."""
        def action() -> None:
            with open(self.path, "a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(str(line).rstrip("\n") + "\n")
                handle.flush()
        self._with_lock(action)

    def summary(self) -> str:
        with self._lock:                        # RLock: safe if called while locked
            size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return f"{os.path.basename(self.path)}: {len(self.read_lines())} lines, {size} bytes"


# ------------------------------------------------------------------ workers --
def run_workers(safe_file: ThreadSafeFile, n_threads: int, n_lines: int,
                prefix: str = "worker", delay: float = 0.0) -> threading.Thread:
    """Start n_threads, each appending n_lines numbered lines."""
    def work(index: int) -> None:
        for n in range(n_lines):
            safe_file.append(f"{prefix}-{index} line {n:03d}")
            if delay:
                time.sleep(delay)              # widen the race window

    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=work, args=(i + 1,), name=f"{prefix}-{i + 1}")
        t.start()
        threads.append(t)
    return threads


def demo_race_condition(path: str, n_threads: int = 8, n_lines: int = 40) -> tuple[int, int]:
    """Write the same file WITHOUT a lock, to show what the lock prevents.

    Returns (expected_lines, actual_lines). actual < expected means writes were
    lost or interleaved - that is the bug the lock fixes.
    """
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def work(index: int) -> None:
        for n in range(n_lines):
            with open(path, "a", encoding="utf-8") as handle:   # no lock at all
                handle.write(f"racy-{index} line {n:03d}\n")

    threads = [threading.Thread(target=work, args=(i + 1,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(path, "r", encoding="utf-8") as handle:
        actual = len(handle.read().splitlines())
    return n_threads * n_lines, actual
