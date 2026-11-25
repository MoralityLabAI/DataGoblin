"""Draw seed-level retention in the dense grid after the final parity stage."""
import argparse
import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor


def draw(source, output):
    rows = json.loads(source.read_bytes())
    methods = ("sequential", "interleaved", "replay", "distill", "project", "route", "heads")
    colors = [HexColor("#195b58"), HexColor("#ab5933")]
    ink = HexColor("#252e38")
    plot = Drawing(640, 305)

    def label(x, y, value, size=10, weight="Helvetica"):
        plot.add(String(x, y, value, fontName=weight, fontSize=size, fillColor=ink))

    plot.add(Rect(0, 0, 640, 305, fillColor=HexColor("#ffffff"), strokeColor=None))
    label(16, 282, "Which switches survive the final training stage?", 17, "Helvetica-Bold")
    label(16, 262, "Fraction of 201 deployment priors with an optimal encoder. One dot per training seed.")
    for j, (d, title) in enumerate((("p3", "Three-bit game"), ("p4", "Four-bit game"))):
        x = 115 + 255 * j
        label(x, 232, title, 12, "Helvetica-Bold")
        for frac in (0, .5, 1):
            xp = x + 190 * frac
            plot.add(Line(xp, 42, xp, 216, strokeWidth=.5, strokeColor=HexColor("#c3cdd1")))
            label(xp - 5, 26, f"{frac:g}")
        for idx, method in enumerate(methods):
            y = 205 - idx * 25
            if j == 0:
                label(16, y - 3, method.capitalize())
            for pos, seed in enumerate((137, 163, 191)):
                r = next(r for r in rows if r["key"] == f"{method}-{seed}-p4")
                xx = x + 190 * r["metrics"][d + "_optimal_fraction"]
                plot.add(Circle(xx, y + 3 * (pos - 1), 2.8, fillColor=colors[j], strokeColor=None))
    output.parent.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(plot, str(output))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    draw(args.source, args.output)
