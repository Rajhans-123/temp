"""Builds the HTML preview of the A4 report.

    python report/make_report.py     -> report/dwm_report.html

All wording, numbers and figures come from report/content.py, the same source the
print deliverable (report/make_docx.py -> report/dwm_report.docx) uses, so the
preview and the Word document can never disagree. Open the HTML in a browser and
Ctrl+P -> Save as PDF (A4, default margins) if you want a PDF preview.
"""
from __future__ import annotations

import os
import re
import sys

# run from anywhere: the project root holds `core`, this folder holds content.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import content as K
import diagrams

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dwm_report.html")

# ------------------------------------------------------------------ styles --
CSS = """
@page { size: A4 portrait; margin: 11mm 11mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Segoe UI", Arial, Helvetica, sans-serif;
  font-size: 8.4pt; line-height: 1.3; color: #1f2937; background: #ffffff;
  margin: 0 auto; max-width: 190mm;
}
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }

h1 { font-size: 15.5pt; line-height: 1.14; margin: 0 0 2.2mm; color: #111827;
     letter-spacing: -0.3px; }
h2 { font-size: 10.8pt; margin: 0 0 1.4mm; color: #4f46e5;
     border-bottom: 1.2pt solid #4f46e5; padding-bottom: 1mm; }
h3 { font-size: 9.2pt; margin: 2mm 0 0.8mm; color: #111827; }
h4 { font-size: 9pt; margin: 1.8mm 0 0.8mm; color: #374151; }
p  { margin: 0 0 1.5mm; text-align: justify; }
ul, ol { margin: 0 0 1.8mm; padding-left: 4.5mm; }
li { margin-bottom: 0.7mm; }
code, .mono {
  font-family: Consolas, "Courier New", monospace; font-size: 7.9pt;
  background: #f3f4f6; padding: 0.3mm 0.8mm; border-radius: 2.5px;
  color: #0f172a;
}
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3pt solid #4f46e5;
      border-radius: 4px; padding: 1.8mm 2.2mm; overflow-x: auto;
      font-family: Consolas, "Courier New", monospace; font-size: 7.6pt;
      line-height: 1.3; color: #0f172a; margin: 0 0 2mm; }
pre code { background: none; padding: 0; font-size: inherit; }

table { width: 100%; border-collapse: collapse; margin: 0 0 2mm;
        font-size: 7.6pt; }
th { background: #eef2ff; color: #312e81; text-align: left; font-weight: 600;
     padding: 1mm 1.4mm; border: 0.5pt solid #c7d2fe; }
td { padding: 0.85mm 1.4mm; border: 0.5pt solid #e5e7eb; vertical-align: top; }
tbody tr:nth-child(even) { background: #fafafa; }
td.num, th.num { text-align: right; font-family: Consolas, monospace; }
.best { background: #dcfce7 !important; font-weight: 600; }

.figure { margin: 0 auto 2.2mm; }
.figure svg { width: 100%; height: auto; border: 0.5pt solid #e5e7eb;
              border-radius: 4px; }
figcaption { font-size: 7.2pt; color: #6b7280; margin-top: 0.9mm; }

.shot {
  border: 1.4pt dashed #94a3b8; border-radius: 5px; background: #f8fafc;
  padding: 2.2mm; text-align: center; margin: 0 0 1mm; height: 32mm;
  display: flex; flex-direction: column; justify-content: center;
}
.shot .tag { font-size: 6.9pt; letter-spacing: 1pt; text-transform: uppercase;
             color: #64748b; margin-bottom: 1.2mm; }
.shot .name { font-size: 9.2pt; font-weight: 600; color: #334155; }
.shot .how { font-size: 7.4pt; color: #64748b; margin-top: 1.2mm; }
.shot-grid { display: flex; gap: 2.4mm; }
.shot-grid > div { flex: 1; }

.kpis { display: flex; gap: 2mm; margin: 0 0 2.4mm; }
.kpi { flex: 1; border: 0.6pt solid #e2e8f0; border-radius: 5px;
       padding: 1.4mm 1.6mm; background: #f8fafc; }
.kpi .v { font-size: 11.5pt; font-weight: 700; color: #4f46e5; line-height: 1.1; }
.kpi .l { font-size: 6.9pt; color: #64748b; text-transform: uppercase;
          letter-spacing: 0.4px; margin-top: 0.5mm; }
.kpi .s { font-size: 6.9pt; color: #94a3b8; }

.callout { border-left: 2.6pt solid #4f46e5; background: #eef2ff;
           padding: 1.4mm 2mm; margin: 0 0 1.8mm; font-size: 8pt; }
.callout.warn { border-left-color: #d97706; background: #fffbeb; }
.callout.ok   { border-left-color: #16a34a; background: #f0fdf4; }
.callout b { color: #1e1b4b; }

.cover { text-align: center; padding-top: 8mm; }
.cover .rule { height: 2.4pt; background: #4f46e5; width: 42mm; margin: 0 auto 7mm; }
.cover h1 { font-size: 21pt; }
.cover .sub { font-size: 10.5pt; color: #4b5563; margin-bottom: 2.4mm; }
.cover .course { font-size: 9.5pt; color: #4f46e5; font-weight: 600;
                 letter-spacing: 1.3pt; text-transform: uppercase; margin-bottom: 9mm; }
.cover table { font-size: 8.4pt; }
.cover td { border: none; padding: 0.8mm 1.4mm; }
.cover td:first-child { color: #64748b; width: 34mm; text-align: right;
                        font-weight: 600; }
.cover td:last-child { text-align: left; }

.toc td { border: none; padding: 0.7mm 0; font-size: 8.6pt; }
.toc td:first-child { width: 8mm; color: #4f46e5; font-weight: 700; }
.toc td:last-child { text-align: right; color: #94a3b8; }
.footer { margin-top: 1.8mm; padding-top: 0.9mm; border-top: 0.5pt solid #e5e7eb;
          font-size: 7pt; color: #94a3b8; display: flex; justify-content: space-between; }
"""


def page(content: str, page_no: int, total: int, title: str) -> str:
    return f"""<div class="page">{content}
  <div class="footer"><span>{title}</span><span>Page {page_no} of {total}</span></div>
</div>"""


def shot(name: str, how: str, tag: str = "Screenshot slot", h: int = 32) -> str:
    return (f'<div class="shot" style="height:{h}mm"><div class="tag">{tag}</div>'
            f'<div class="name">{name}</div>'
            f'<div class="how">{how}</div></div>')


def figure(key: str, caption: str, width: int = 100) -> str:
    """`width` shrinks the diagram (and its height, proportionally) to fit a page."""
    return (f'<div class="figure" style="width:{width}%">{diagrams.FIGURES[key]()}'
            f'<figcaption>{caption}</figcaption></div>')


def table(headers: list[str], rows: list[list], opts: dict | None = None) -> str:
    opts = opts or {}
    numeric = opts.get("numeric", set())
    code_cols = opts.get("code_cols", set())
    best_row = opts.get("best")
    style = (f' style="font-size:{opts["font"]}pt"' if "font" in opts else "")
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{h}</th>'
        for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        cls = "best" if best_row is not None and r == best_row else ""
        cells = ""
        for i, c in enumerate(row):
            text = f"<code>{c}</code>" if i in code_cols else str(c)
            klass = "num " + cls if i in numeric else cls
            cells += f'<td class="{klass}">{text}</td>'
        body += f"<tr class='{cls}'>{cells}</tr>"
    return (f"<table{style}><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>")


def kpis(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="kpi"><div class="v">{v}</div>'
                    f'<div class="l">{l}</div><div class="s">{s}</div></div>'
                    for v, l, s in items)
    return f'<div class="kpis">{cells}</div>'


# ------------------------------------------------------------ blocks -------
_TOKEN = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")


def _markup(piece: str) -> str:
    if piece.startswith("`"):
        return f"<code>{piece[1:-1]}</code>"
    if piece.startswith("**"):
        return f"<b>{piece[2:-2]}</b>"
    return f"<i>{piece[1:-1]}</i>"


def rich(text: str) -> str:
    """Escapes the text, then applies the shared inline markup."""
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;"))
    return "".join(_markup(p) if _TOKEN.fullmatch(p) else p
                   for p in _TOKEN.split(escaped))


def cover(spec: dict) -> str:
    kpi_row = kpis(spec["kpis"])
    titles = "<br>".join(t for t in spec["title"].split("\n"))
    meta = "".join(
        f"<tr><td>{key}</td><td>{rich(value)}</td></tr>"
        for key, value in spec["rows"])
    return f"""
<div class="cover">
  <div class="course">{rich(spec["course"])}</div>
  <div class="rule"></div>
  <h1>{titles}</h1>
  <div class="sub">{rich(spec["subtitle"])}</div>
  <p style="text-align:center;max-width:150mm;margin:6mm auto 10mm">{rich(spec["blurb"])}</p>
  {kpi_row}
  <table style="margin-top:6mm">{meta}</table>
</div>"""


def toc(rows: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f"<tr><td>{number}</td><td>{rich(title)}</td><td>Page {pages}</td></tr>"
        for number, title, pages in rows)
    return f'<table class="toc">{cells}</table>'


def render_block(block: tuple) -> str:
    kind = block[0]
    if kind == "cover":
        return cover(block[1])
    if kind == "toc":
        return toc(block[1])
    if kind in ("h1", "h2", "h3"):
        return f"<{kind}>{rich(block[1])}</{kind}>"
    if kind == "p":
        return f"<p>{rich(block[1])}</p>"
    if kind == "pre":
        escaped = block[1].replace("&", "&amp;").replace("<", "&lt;")
        return f"<pre><code>{escaped}</code></pre>"
    if kind == "callout":
        cls = f' callout-{block[1]}' if block[1] else ""
        return f'<div class="callout{cls}">{rich(block[2])}</div>'
    if kind == "bullets":
        return "<ul>" + "".join(f"<li>{rich(i)}</li>" for i in block[1]) + "</ul>"
    if kind == "numbered":
        return "<ol>" + "".join(f"<li>{rich(i)}</li>" for i in block[1]) + "</ol>"
    if kind == "table":
        return table(block[1], block[2], block[3])
    if kind == "figure":
        return figure(block[1], rich(block[2]), block[3])
    if kind == "shots":
        inner = "".join(
            f'<div>{shot(name, how, h=int(height))}</div>'
            for name, how, height in block[1])
        return f'<div class="shot-grid">{inner}</div>'
    raise ValueError(f"unknown block {kind!r}")


def build() -> str:
    total = len(K.PAGES)
    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>DWM Anime Mining Pipeline - Project Report</title>",
            f"<style>{CSS}</style></head><body>"]
    for page_no, blocks in enumerate(K.PAGES, 1):
        body = "".join(render_block(block) for block in blocks)
        html.append(page(body, page_no, total, K.PAGE_TITLES[page_no - 1]))
    html.append("</body></html>")
    return "".join(html)


if __name__ == "__main__":
    document = build()
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"written: {OUT}  ({len(document) / 1024:.0f} KB)")
    print("preview: open in a browser. Print deliverable: python report/make_docx.py")
