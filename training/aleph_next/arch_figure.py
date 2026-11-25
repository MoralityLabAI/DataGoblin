"""Plot the rule-family-balanced verification rates from frozen architecture receipts."""

import argparse
import json
from pathlib import Path
from statistics import mean

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors


def figure(directory, path):
    scores = json.loads((directory / "scores.json").read_text())["native"]
    config = json.loads((directory / "training.json").read_text())["contract"]["recipe"]
    seeds = config["seeds"]
    groups = (
        ("direct2", "source2", "local2", "gate_source2", "gate_local2", "damp2"),
        ("direct3", "rlt3", "damp3", "magnitude3", "visit3"),
    )
    labels = {
        "direct2": "Direct",
        "source2": "Source",
        "local2": "Local",
        "gate_source2": "Learned source",
        "gate_local2": "Learned local",
        "damp2": "Damping / RLT",
        "direct3": "Direct",
        "rlt3": "RLT",
        "damp3": "Damping",
        "magnitude3": "Magnitude clock",
        "visit3": "Visit clock",
    }
    palette = ["#172B4D", "#007A7B", "#BF5B18", "#6F52A1", "#A43B65", "#526E37"]
    drawing = Drawing(640, 545)
    drawing.add(
        String(
            22,
            521,
            "Verified repairs and counterexamples under hidden-state corruption",
            fontName="Helvetica-Bold",
            fontSize=15,
        )
    )
    drawing.add(
        String(
            22,
            501,
            f"Mean success across held rule compositions, task types and {len(seeds)} matched seeds.",
            fontName="Helvetica",
            fontSize=10,
        )
    )
    for p, arms in enumerate(groups):
        y0 = 293 - 230 * p
        drawing.add(String(22, y0 + 181, ("Two visits", "Three visits")[p], fontName="Helvetica-Bold", fontSize=12))
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 63, y0 + 13, 370, 145
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, 1.25
        chart.xValueAxis.valueSteps = [0, 0.5, 1.25]
        chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = 0, 100
        chart.yValueAxis.valueStep = 20
        chart.yValueAxis.visibleGrid = True
        chart.yValueAxis.gridStrokeColor = colors.HexColor("#D5DCE1")
        chart.xValueAxis.labels.fontSize = 9
        chart.yValueAxis.labels.fontSize = 9
        lines = []
        used = [a for a in arms if a in config["arms"]]
        for j, a in enumerate(used):
            values = []
            for strength, conditions in (
                (0, ("clean-0.0",)),
                (0.5, ("target-0.5", "random-0.5")),
                (1.25, ("target-1.25", "random-1.25")),
            ):
                v = mean(
                    row["verified"][0]
                    for seed in seeds
                    for kind in conditions
                    for row in scores[f"{a}-{seed}"][kind].values()
                )
                values.append((strength, v * 100))
            lines.append(values)
            color = colors.HexColor(palette[j])
            chart.lines[j].strokeColor = color
            chart.lines[j].strokeWidth = 1.8
            chart.lines[j].symbol = makeMarker("FilledCircle", size=4, fillColor=color)
            drawing.add(
                String(468, y0 + 147 - j * 18, labels[a], fontName="Helvetica-Bold", fontSize=10, fillColor=color)
            )
        chart.data = lines
        drawing.add(chart)
        drawing.add(String(64, y0 - 14, "Injected RMS / incoming-state RMS", fontName="Helvetica", fontSize=9))
        drawing.add(String(22, y0 + 158, "%", fontName="Helvetica", fontSize=9))
    drawing.add(
        String(
            22,
            20,
            "Nonzero levels average targeted and equal-magnitude random directions. Four positions per proof table.",
            fontName="Helvetica",
            fontSize=9,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(drawing, str(path))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    figure(args.directory, args.output)
