"""Renders report/diagrams.py SVG output to PNG using headless Edge.

    python report/render_figures.py

python-docx cannot embed SVG, so the Word report needs raster figures. Each SVG is
placed in a bare HTML page at a fixed pixel width and screenshotted at 2x device
scale, which keeps the text in the boxes legible when the figure is printed at
16-17 cm wide.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diagrams  # noqa: E402

TMP = r"C:\Users\deeks\AppData\Local\Temp\opencode"
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PX_WIDTH = 1500          # render width; inserted into Word at 16.5 cm


def find_edge() -> str:
    for path in EDGE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit("no Chromium-based browser found for SVG -> PNG rendering")


def render(only: list[str] | None = None) -> list[str]:
    edge = find_edge()
    os.makedirs(FIG_DIR, exist_ok=True)
    written = []
    for key, builder in diagrams.FIGURES.items():
        if only and key not in only:
            continue
        svg = builder()
        width = PX_WIDTH
        height = int(round(width * _aspect(key)))
        html = (f"<!doctype html><html><head><meta charset='utf-8'><style>"
                f"html,body{{margin:0;padding:0;background:#fff;width:{width}px}}"
                f"svg{{display:block;width:{width}px;height:{height}px}}"
                f"</style></head><body>{svg}</body></html>")
        html_path = os.path.join(TMP, f"fig_{key}.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        png = os.path.join(FIG_DIR, f"fig_{key}.png")
        if os.path.exists(png):
            os.remove(png)
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={width},{height}",
                        f"--screenshot={png}", f"file:///{html_path.replace(os.sep, '/')}"],
                       capture_output=True, timeout=120)
        if os.path.exists(png):
            written.append(png)
            print(f"  {os.path.basename(png):24s} {os.path.getsize(png) / 1024:7.0f} KB")
        else:
            print(f"  FAILED: {key}")
    return written


def _aspect(key: str) -> float:
    """height / width of each diagram, taken from the Diagram defaults."""
    sizes = {"end_to_end": 430 / 980, "preprocessing": 350 / 940, "j48": 330 / 940,
             "clustering": 330 / 940, "prediction": 360 / 940, "ui": 300 / 940}
    return sizes[key]


if __name__ == "__main__":
    targets = sys.argv[1:] or None
    print(f"rendering diagrams with {os.path.basename(find_edge())}")
    out = render(targets)
    print(f"{len(out)} figure(s) written to {FIG_DIR}")