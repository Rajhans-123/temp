"""SVG flowchart helpers - small, dependency-free, print-friendly (white bg).

Every diagram in the report is generated here, so the shapes, arrows and labels
are real vector graphics rather than hand-drawn approximations.
"""
from __future__ import annotations

# VS Code light-theme palette so figures sit naturally next to the code pages
INK = "#1f2937"
MUTED = "#6b7280"
LINE = "#4f46e5"
FILL_INPUT = "#eef2ff"
FILL_PROC = "#ecfdf5"
FILL_STORE = "#fef3c7"
FILL_OUT = "#f1f5f9"
FILL_ERROR = "#fee2e2"

_FONT = ("Segoe UI, Arial, Helvetica, sans-serif")

# The SVG is printed at ~14 cm wide, where a 10-unit label would land at 4 pt.
# Every font size is multiplied by this factor at emission time so the smallest
# printed text stays above ~5.5 pt; wrap widths shrink to match so labels still
# fit their boxes.
FONT_SCALE = 1.32
_TEXT_W = 0.56   # average glyph width in em (Segoe UI), used for fit checks


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width or not current:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class Diagram:
    """Accumulates SVG shapes and renders them to a string."""

    def __init__(self, width: int = 900, height: int = 400, title: str = ""):
        self.width = width
        self.height = height
        self.parts: list[str] = []

    # ------------------------------------------------------------- shapes --
    def box(self, x: int, y: int, w: int, h: int, label: str, *,
            fill: str = FILL_PROC, stroke: str = LINE, dashed: bool = False,
            radius: int = 6, bold: bool = False, fontsize: int = 12,
            sublabel: str = "", wrap_at: int = 20, sub_wrap_at: int | None = None,
            ) -> None:
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')

        # Grow the font for print legibility, then shrink it back only as far as
        # this box needs, so tall stacks and long words never spill their frame.
        base, pad = fontsize * FONT_SCALE, 7.0
        fs, lines, sub_lines = base, [label], []
        for _ in range(8):
            label_w = max(8, int(wrap_at * fontsize / fs))
            sub_w = max(8, int((sub_wrap_at or wrap_at) * fontsize / fs))
            lines = _wrap(label, label_w)
            sub_lines = _wrap(sublabel, sub_w) if sublabel else []
            stack = lines + sub_lines
            line_height = fs + 3
            need_h = (len(stack) - 1) * line_height + fs * 0.85
            need_w = max(len(s) for s in stack) * _TEXT_W * fs
            shrink = 1.0
            if need_h > h - pad * 2:
                shrink = min(shrink, (h - pad * 2) / need_h)
            if need_w > w - pad * 2:
                shrink = min(shrink, (w - pad * 2) / need_w)
            if shrink > 0.995 or fs <= fontsize:
                break
            fs = round(max(fontsize, fs * shrink), 1)

        stack = lines + sub_lines
        total = len(stack)
        line_height = fs + 3
        start = y + h / 2 - (total - 1) * line_height / 2 + fs * 0.34
        for i, line in enumerate(stack):
            weight = "600" if bold or i == 0 else "400"
            colour = MUTED if sub_lines and i >= len(lines) else INK
            self.parts.append(
                f'<text x="{x + w / 2}" y="{start + i * line_height}" '
                f'font-family="{_FONT}" font-size="{fs:g}" font-weight="{weight}" '
                f'fill="{colour}" text-anchor="middle">{_esc(line)}</text>')

    def diamond(self, x: int, y: int, w: int, h: int, label: str,
                fontsize: int = 11) -> None:
        self.parts.append(
            f'<polygon points="{x + w / 2},{y} {x + w},{y + h / 2} '
            f'{x + w / 2},{y + h} {x},{y + h / 2}" '
            f'fill="#f5f3ff" stroke="{LINE}" stroke-width="1.6"/>')
        fs = round(fontsize * FONT_SCALE, 1)
        for _ in range(8):
            stack = _wrap(label, max(8, int(16 * fontsize / fs)))
            need_h = (len(stack) - 1) * (fs + 3) + fs * 0.85
            need_w = max(len(s) for s in stack) * _TEXT_W * fs
            shrink = 1.0
            if need_h > h * 0.66:
                shrink = min(shrink, h * 0.66 / need_h)
            if need_w > w * 0.76:
                shrink = min(shrink, w * 0.76 / need_w)
            if shrink > 0.995 or fs <= fontsize:
                break
            fs = round(max(fontsize, fs * shrink), 1)
        start = y + h / 2 - (len(stack) - 1) * (fs + 3) / 2 + fs * 0.34
        for i, line in enumerate(stack):
            self.parts.append(
                f'<text x="{x + w / 2}" y="{start + i * (fs + 3)}" '
                f'font-family="{_FONT}" font-size="{fs:g}" fill="{INK}" '
                f'text-anchor="middle">{_esc(line)}</text>')

    def arrow(self, x1: int, y1: int, x2: int, y2: int, label: str = "",
              dashed: bool = False, colour: str = LINE) -> None:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
            f'stroke-width="1.7"{dash} marker-end="url(#arrowhead)"/>')
        if label:
            self.parts.append(
                f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" '
                f'font-family="{_FONT}" font-size="{10.5 * FONT_SCALE:g}" fill="{MUTED}" '
                f'text-anchor="middle">{_esc(label)}</text>')

    def elbow(self, x1: int, y1: int, x2: int, y2: int, via_y: int | None = None,
              label: str = "", colour: str = LINE, dashed: bool = False) -> None:
        """Right-angle connector - cleaner than diagonals for pipelines."""
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        mid = via_y if via_y is not None else (y1 + y2) / 2
        points = f"{x1},{y1} {x1},{mid} {x2},{mid} {x2},{y2}"
        self.parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="1.7"{dash} marker-end="url(#arrowhead)"/>')
        if label:
            self.parts.append(
                f'<text x="{(x1 + x2) / 2}" y="{mid - 5}" font-family="{_FONT}" '
                f'font-size="{10.5 * FONT_SCALE:g}" fill="{MUTED}" text-anchor="middle">'
                f'{_esc(label)}</text>')

    def label(self, x: int, y: int, text: str, size: int = 11,
              colour: str = MUTED, anchor: str = "start", bold: bool = False) -> None:
        size = round(size * FONT_SCALE, 1)
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{_FONT}" font-size="{size}" '
            f'fill="{colour}" text-anchor="{anchor}" '
            f'font-weight="{"600" if bold else "400"}">{_esc(text)}</text>')

    def group_box(self, x: int, y: int, w: int, h: int, title: str) -> None:
        """Dashed container around a group of boxes."""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" '
            f'stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.7"/>')
        self.label(x + 10, y + 16, title, size=11, colour=MUTED, bold=True)

    # ------------------------------------------------------------- render --
    def render(self) -> str:
        defs = (
            '<defs><marker id="arrowhead" markerWidth="9" markerHeight="9" '
            'refX="7.5" refY="3" orient="auto">'
            f'<path d="M0,0 L0,6 L7.5,3 z" fill="{LINE}"/>'
            "</marker></defs>")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" '
                f'viewBox="0 0 {self.width} {self.height}" '
                f'style="max-width:100%;height:auto;background:#ffffff" '
                f'font-family="{_FONT}">{defs}{"".join(self.parts)}</svg>')


# ===========================================================================
#  The seven diagrams used in the report
# ===========================================================================
def fig_end_to_end() -> str:
    d = Diagram(980, 430)
    d.label(20, 24, "Figure 1  End-to-end data flow of the project", 13, INK, bold=True)

    d.box(20, 60, 175, 74, "Raw CSV", fill=FILL_INPUT,
          sublabel="anime.csv  12,294 rows", wrap_at=24)
    d.box(20, 155, 175, 74, "ratings.csv", fill=FILL_INPUT,
          sublabel="7,813,737 rows", wrap_at=24)
    d.arrow(196, 97, 250, 130)
    d.arrow(196, 192, 250, 160)

    d.box(255, 108, 190, 96, "Stage 1  Preprocessing", bold=True, wrap_at=18,
          sublabel="clean · impute · encode · log · scale · bin · split")
    d.arrow(446, 156, 500, 156)

    d.group_box(500, 42, 250, 232, "Mining stages (core/*.py)")
    d.box(516, 68, 218, 44, "2  Association rules", wrap_at=24, fontsize=11)
    d.box(516, 120, 218, 44, "3  Classification J48 / NB", wrap_at=24, fontsize=11)
    d.box(516, 172, 218, 44, "4  Regression", wrap_at=24, fontsize=11)
    d.box(516, 224, 44, 40, "5", fontsize=13, wrap_at=2)
    d.box(562, 224, 82, 40, "Clustering", wrap_at=10, fontsize=11)
    d.box(646, 224, 88, 40, "Prediction", wrap_at=10, fontsize=11)
    for y in (90, 142, 194, 244):
        d.arrow(450, 156, 516, y, dashed=True, colour=MUTED)

    d.arrow(752, 156, 800, 156)
    d.box(805, 60, 155, 76, "CSV reports", fill=FILL_STORE, wrap_at=18,
          sublabel="reports/*.csv")
    d.box(805, 150, 155, 76, "Trained models", fill=FILL_STORE, wrap_at=18,
          sublabel="models/*.joblib")
    d.box(805, 240, 155, 76, "Streamlit UI", fill=FILL_OUT, wrap_at=18,
          sublabel="app.py  7 tabs")
    d.elbow(752, 244, 790, 244, via_y=278)
    d.elbow(882, 250, 882, 235)

    d.label(20, 400, "Every arrow writes CSV. No database server, no JSON "
                     "artefacts - the whole pipeline is inspectable.", 10.5)
    return d.render()


def fig_preprocessing() -> str:
    d = Diagram(940, 350)
    d.label(20, 24, "Figure 2  Stage 1 - what happens to every row of anime.csv", 13,
            INK, bold=True)
    boxes = [
        ("Load 12,294 rows", "profile dtype, nulls, duplicates", FILL_INPUT),
        ("Deduplicate", "drop duplicate anime_id", FILL_PROC),
        ("Impute", "genre/type -> Unknown\nrating -> type mean\nepisodes -> type median", FILL_PROC),
        ("Feature engineer", "log1p(members),\nlog1p(episodes),\ngenre flags + count",
         FILL_PROC),
        ("Outlier flag", "1.5 x IQR fence\nkeep + set is_outlier", FILL_PROC),
        ("Scale", "min-max and z-score\n(needed by clustering)", FILL_PROC),
        ("Discretise", "rating -> Low/Medium/\nHigh/Top + member bands", FILL_PROC),
        ("Split 80/20", "stratified, seed 42", FILL_OUT),
    ]
    x, w, gap = 20, 105, 12
    for i, (title, sub, fill) in enumerate(boxes):
        y = 60 if i % 2 == 0 else 165
        d.box(x, y, w, 88, title, fill=fill, fontsize=10.5, wrap_at=13,
              sublabel=sub, sub_wrap_at=15)
        if i:
            d.elbow(x - gap, 104 if (i - 1) % 2 == 0 else 209, x, y + 44,
                    via_y=104 if (i - 1) % 2 == 0 else 209)
        x += w + gap
    d.box(20, 280, 300, 52, "anime_clean.csv  (master table)", fill=FILL_STORE,
          fontsize=10.5, wrap_at=30)
    d.box(330, 280, 300, 52, "anime_rating_stats.csv  (per anime)", fill=FILL_STORE,
          fontsize=10.5, wrap_at=30)
    d.box(640, 280, 280, 52, "user_features.csv  (per user)", fill=FILL_STORE,
          fontsize=10.5, wrap_at=30)
    d.elbow(600, 253, 170, 280, via_y=268)
    d.elbow(640, 253, 480, 280, via_y=268)
    d.elbow(700, 253, 780, 280, via_y=268)
    return d.render()


def fig_j48() -> str:
    d = Diagram(940, 330)
    d.label(20, 24, "Figure 3  Stage 3 - how the J48 (C4.5) classifier decides", 13,
            INK, bold=True)

    d.box(30, 60, 150, 66, "Training frame", fill=FILL_INPUT, wrap_at=16,
          sublabel="9,513 rows x 12 attributes", fontsize=10.5)
    d.arrow(182, 93, 232, 93)

    d.box(236, 52, 170, 82, "Pre-discretise numerics", bold=True, wrap_at=18,
          sublabel="members, episodes -> 10 quantile bins")
    d.arrow(408, 93, 458, 93)

    d.box(462, 52, 150, 82, "Gain ratio split", fill=FILL_PROC, wrap_at=16,
          sublabel="gain / split-info\nmax at each node", fontsize=10.5)
    d.arrow(614, 93, 664, 93)

    d.diamond(668, 55, 110, 76, "pure node\nor depth limit?")
    d.arrow(780, 93, 826, 93, label="yes")
    d.box(830, 60, 95, 66, "Leaf = class", fill=FILL_OUT, wrap_at=12,
          sublabel="majority vote", fontsize=10)
    d.elbow(723, 133, 340, 175, via_y=160, label="no - keep splitting")

    d.box(30, 178, 150, 66, "Post-prune", fill=FILL_PROC, wrap_at=16,
          sublabel="confidence factor\nbottom-up", fontsize=10.5)
    d.arrow(182, 211, 232, 211)
    d.box(236, 178, 170, 66, "Extract rules", fill=FILL_PROC, wrap_at=16,
          sublabel="IF ... THEN class", fontsize=10.5)
    d.arrow(408, 211, 458, 211)

    d.box(462, 168, 240, 86, "Measured on 2,379 held-out rows", fill=FILL_OUT,
          bold=True, wrap_at=22, sublabel="accuracy 0.5595  F1 0.5291")
    d.arrow(704, 211, 754, 211)
    d.box(758, 168, 167, 86, "reports/j48_rules.csv", fill=FILL_STORE, wrap_at=18,
          sublabel="5 levels, 27 rules")

    d.label(30, 294, "Naive Bayes runs on the same features: Gaussian NB on the "
                     "scaled numeric columns,", 10.5)
    d.label(30, 310, "Categorical NB on the one-hot categoricals.", 10.5)
    return d.render()


def fig_clustering() -> str:
    d = Diagram(940, 330)
    d.label(20, 24, "Figure 4  Stage 5 - two datasets, two algorithms, "
                    "one shared workflow", 13, INK, bold=True)

    d.box(20, 70, 160, 60, "Dataset A", fill=FILL_INPUT, wrap_at=14,
          sublabel="anime_clean.csv", fontsize=11)
    d.box(20, 200, 160, 60, "Dataset B", fill=FILL_INPUT, wrap_at=14,
          sublabel="user_features.csv", fontsize=11)
    d.arrow(182, 100, 236, 130)
    d.arrow(182, 230, 236, 200)
    d.box(240, 118, 150, 92, "StandardScaler", fill=FILL_PROC, wrap_at=14,
          sublabel="z-score, drop nulls", fontsize=10.5)
    d.arrow(392, 164, 440, 164)
    d.box(444, 118, 155, 92, "Choose k", fill=FILL_PROC, wrap_at=14,
          sublabel="elbow + silhouette\n+ CH, DB index", fontsize=10.5)

    d.elbow(601, 145, 640, 105, via_y=105, label="K-Means")
    d.elbow(601, 185, 640, 215, via_y=240, label="Ward")
    d.box(644, 66, 160, 78, "K-Means", fill=FILL_PROC, wrap_at=14,
          sublabel="n_init=10, seed 42", fontsize=10.5)
    d.box(644, 206, 160, 78, "Agglomerative (Ward)", fill=FILL_PROC, wrap_at=16,
          sublabel="1,200-row sample,\ndendrogram kept", fontsize=10.5)

    d.elbow(806, 105, 846, 145, via_y=145)
    d.elbow(806, 245, 846, 205, via_y=205)
    d.box(850, 130, 80, 92, "Profile", fill=FILL_STORE, wrap_at=9, fontsize=10,
          sublabel="centroid\nmeans +\nauto name")
    d.label(20, 310, "Silhouette: A = 0.341 (K-Means) / 0.310 (Ward) at k=4   ·   "
                     "B = 0.276 / 0.220 at k=5", 10.5)
    return d.render()


def fig_prediction() -> str:
    d = Diagram(940, 360)
    d.label(20, 24, "Figure 5  Stage 6 - the two models the UI actually calls", 13,
            INK, bold=True)

    d.group_box(20, 44, 440, 296, "A. Score predictor")
    d.box(38, 72, 180, 62, "Form input", fill=FILL_INPUT, wrap_at=14,
          sublabel="type, episodes, members, genres", fontsize=10, sub_wrap_at=15)
    d.arrow(220, 103, 268, 103)
    d.box(272, 72, 170, 62, "Build feature row", fill=FILL_PROC, wrap_at=16,
          sublabel="same transforms as training", fontsize=10)
    d.box(38, 158, 180, 54, "Random Forest", fill=FILL_PROC, wrap_at=13, fontsize=10.5)
    d.box(232, 158, 180, 54, "Gradient Boosting", fill=FILL_PROC, wrap_at=16,
          fontsize=10.5)
    d.elbow(300, 134, 130, 158, via_y=146)
    d.arrow(330, 134, 330, 154)
    d.box(38, 250, 180, 58, "Weighted score", fill=FILL_OUT, wrap_at=14,
          sublabel="0.42 RF + 0.58 GB", fontsize=10)
    d.arrow(128, 212, 128, 248)
    d.elbow(322, 212, 180, 248, via_y=230)
    d.arrow(220, 279, 268, 279)
    d.box(272, 250, 170, 58, "Tier + 95% interval", fill=FILL_OUT, wrap_at=16,
          sublabel="+/- 1.96 x 0.653", fontsize=10)

    d.group_box(490, 44, 430, 296, "B. Hybrid recommender")
    d.box(508, 72, 190, 56, "K-Means user segments", fill=FILL_INPUT, wrap_at=17,
          sublabel="5 segments, 47,153 users", fontsize=10)
    d.box(716, 72, 186, 56, "Segment affinity", fill=FILL_PROC, wrap_at=15,
          sublabel="genre + media type", fontsize=10)
    d.arrow(700, 100, 714, 100)
    d.box(508, 152, 190, 56, "Association rules", fill=FILL_PROC, wrap_at=15,
          sublabel="lift boost if fired", fontsize=10)
    d.box(716, 152, 186, 56, "Community prior", fill=FILL_PROC, wrap_at=14,
          sublabel="rating + popularity", fontsize=10)
    d.arrow(700, 180, 714, 180)
    d.box(508, 234, 394, 60, "Weighted match score", fill=FILL_OUT, wrap_at=24,
          bold=True,
          sublabel="0.30 genre + 0.18 type + 0.20 prior + 0.18 type-dev + 0.14 pop")
    d.elbow(600, 130, 600, 150, label="")
    d.elbow(810, 130, 810, 150)
    d.elbow(603, 212, 650, 234)
    d.elbow(809, 212, 760, 234)
    return d.render()


def fig_ui() -> str:
    d = Diagram(940, 300)
    d.label(20, 24, "Figure 6  How the Streamlit UI is wired to the mining code", 13,
            INK, bold=True)
    d.box(300, 60, 340, 56, "app.py  -  7 tabs", fill=FILL_OUT, bold=True,
          wrap_at=26)
    tabs = [
        ("1 Overview", "reads mining_summary.csv"),
        ("2 Preprocessing", "recomputes + charts"),
        ("3 Association", "re-mines on slider move"),
        ("4 Classification", "trains J48 on slider move"),
        ("5 Regression", "6 regressors + diagnostics"),
        ("6 Clustering", "k / seed / sample sliders"),
        ("7 Prediction", "loads the 2 joblib models"),
    ]
    x = 20
    for title, sub in tabs:
        d.box(x, 165, 124, 66, title, fill=FILL_INPUT, fontsize=10,
              wrap_at=15, sublabel=sub, sub_wrap_at=20)
        d.elbow(470, 116, x + 62, 165, via_y=140)
        x += 130
    d.box(20, 252, 900, 40, "@st.cache_data  -  each tab is computed once per "
          "session, then served from cache", fill=FILL_PROC, wrap_at=100,
          fontsize=10.5)
    d.arrow(470, 231, 470, 250)
    return d.render()


FIGURES = {
    "end_to_end": fig_end_to_end,
    "preprocessing": fig_preprocessing,
    "j48": fig_j48,
    "clustering": fig_clustering,
    "prediction": fig_prediction,
    "ui": fig_ui,
}