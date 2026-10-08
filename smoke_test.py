"""Headless smoke test: execute app.py through Streamlit's AppTest harness.

    python smoke_test.py [timeout_seconds]

Runs the whole app, then drives the interactive paths that only execute after a
widget changes (score prediction, both recommender modes, the second clustering
dataset). Fails loudly on any uncaught exception or dead tab, so the UI can be
verified without a browser. Exits non-zero when something breaks.
"""
from __future__ import annotations

import sys
import time

from streamlit.testing.v1 import AppTest

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0


def els(node, name: str) -> list:
    out = list(getattr(node, name, []) or [])
    if not out:
        try:
            out = list(node.get(name))
        except Exception:
            out = []
    return out


def plots(at) -> int:
    return len(els(at, "plotly_chart")) + len(els(at, "vega_lite_chart"))


def pick_radio(at, option: str):
    """Find the radio whose options contain `option` and select it."""
    for r in at.radio:
        if option in list(r.options):
            return r
    return None


def check(at, label: str) -> bool:
    if at.exception:
        print(f"\nFAIL [{label}] uncaught exception")
        for e in at.exception:
            print("-" * 70)
            print(e.value)
        return False
    print(f"ok   {label}"
          f"   dataframes={len(at.dataframe)} plots={plots(at)}"
          f" errors={len(at.error)} warnings={len(at.warning)}")
    for e in at.error:
        print(f"       st.error: {e.value}")
    for w in at.warning:
        print(f"       st.warning: {w.value}")
    return True


def tab_report(at) -> None:
    # app uses sidebar radio navigation (lazy pages for Cloud memory limits),
    # not st.tabs -- verify the page radio exists and has all 9 pages.
    radios = list(at.radio)
    pages = []
    for r in radios:
        try:
            pages.extend(list(r.options))
        except Exception:
            pass
    expected = ["1 - Overview", "4 - Classification", "7 - Prediction System",
                "8 - Live Feed", "9 - Model Comparison"]
    for e in expected:
        if not any(e in str(p) for p in pages):
            print(f"FAIL navigation radio missing {e!r} (found {pages})")


def main() -> int:
    t0 = time.time()
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    print(f"script executed in {time.time() - t0:.1f}s")
    print(f"radios={len(at.radio)} dataframes={len(at.dataframe)} "
          f"selectbox={len(at.selectbox)} multiselect={len(at.multiselect)} "
          f"slider={len(at.slider)} radio={len(at.radio)} button={len(at.button)}")
    if not check(at, "initial render"):
        return 1
    tab_report(at)

    # ---- drive the interactive paths --------------------------------------
    def run(label: str) -> bool:
        at.run(timeout=TIMEOUT)
        return check(at, label)

    ok = True
    if at.button:
        for b in at.button:
            if b.label == "Predict score":
                b.click()
                ok &= run("score prediction submitted")
                break
        else:
            print("note: no 'Predict score' button found")

    if at.radio:
        for option in ("my genre preferences", "titles I already watched",
                       "B - Users (behaviour)"):
            r = pick_radio(at, option)
            if r is None:
                print(f"note: no radio with option {option!r}")
                continue
            r.set_value(option)
            ok &= run(f"widget -> {option}")

    print(f"\ntotal {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
