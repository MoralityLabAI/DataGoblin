"""Join reproducible decoder fit records and safe aggregate measurements for the Aleph atlas."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atlas_rows(out):
    manifest = out / "training.json"
    train, scores, results = read(manifest), read(out / "fit-scores.json"), read(out / "results.json")
    if sha256(manifest) != results["training_manifest_sha256"] or set(scores) != set(train["fits"]):
        raise ValueError("incompatible training and evaluation artifacts")
    rows = []
    for label, fit in sorted(train["fits"].items()):
        reader, condition, name = fit["reader"], fit["condition"], fit["sender"]
        side = scores[f"{name}-to-{reader}-side"]["mean"]
        checkpoint = out / (label + (".json" if reader == "empirical" else ".pt"))
        if sha256(checkpoint) != fit["sha256"]:
            raise ValueError(f"checkpoint digest changed for {label}")
        metrics = scores[label]["mean"].copy()
        learned = side["brier"] - metrics["brier"]
        oracle = side["bayes"] - metrics["bayes"]
        metrics.update(
            side_matched_learned_value=learned,
            side_matched_oracle_value=oracle,
            recovered_fraction=learned / oracle if oracle > 0.005 else None,
        )
        arm = name.split("-")[0]
        pair = results["comparisons"][f"{arm}-to-{reader}"]
        failures = []
        if condition == "valid":
            if learned <= 0:
                failures.append("Valid upload failed to improve over the matched side-only fit at this seed.")
            if not pair["passed"]:
                failed = [key for key, passed in pair["gates"].items() if not passed]
                failures.append("Three-seed pairing missed " + ", ".join(failed) + ".")
        failures.append("Evaluation covers four fixed nested-mask strata and four independent panel draws.")
        is_table = reader == "empirical"
        rows.append(
            {
                "key": label,
                "checkpoint": checkpoint.name,
                "sha256": fit["sha256"],
                "architecture": reader,
                "params": fit["params"],
                "passes": fit["recurrence"] or 1,
                "seed": fit["seed"],
                "steps": fit["steps"],
                "metrics": metrics,
                "mechanism": {
                    "gym": "ALEPH-03",
                    "condition": condition,
                    "sender": name,
                    "sender_sha256": train["senders"][name]["sha256"],
                    "sender_frozen": True,
                    "parameter_unit": fit["parameter_unit"],
                    "frequency_fit": is_table,
                    "neural_recurrence_visits": fit["recurrence"],
                    "training_corpus": "24 v2 disjoint/overlap native LDT training frames (generator seed 23017)",
                    "recipe": "smoothed hierarchical empirical frequencies"
                    if is_table
                    else "500 one-frame Adam steps, learning rate .003, gradient norm cap 1",
                    "oracle": "per-frame enumeration of eight worlds, checked against native v2 partition risk",
                    "cmi_bits": metrics["cmi_bits"],
                    "structured_prior": "two-mask survival and uniform hypothetical worlds"
                    if reader == "structured"
                    else None,
                },
                "failures": " ".join(failures),
            }
        )
    if len(rows) != 216:
        raise ValueError("incomplete frozen tournament")
    return rows


def plot(summary, path):
    from reportlab.graphics import renderSVG
    from reportlab.graphics.shapes import Circle, Drawing, Line, String
    from reportlab.lib import colors

    data = summary["comparisons"]
    size_x, size_y = 915, 365
    drawing = Drawing(size_x, size_y)
    ink, mute = colors.HexColor("#172838"), colors.HexColor("#516371")
    blue, red = colors.HexColor("#006E76"), colors.HexColor("#AB4828")
    rows = ("C", "D", "MLP", "logistic", "empirical", "structured")

    def label(x, y, value, *, size=11, bold=False, align="start", color=ink):
        drawing.add(
            String(
                x,
                y,
                value,
                fontName="Helvetica-Bold" if bold else "Helvetica",
                fontSize=size,
                textAnchor=align,
                fillColor=color,
            )
        )

    def line(x1, y1, x2, y2, color, width=1):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))

    extent = max(abs(x) for row in data.values() for x in (*row["paired_seed_panel_bootstrap_95"], row["oracle_value"]))
    tick = 0.01
    while extent / tick > 4:
        tick *= 2
    limit = max(tick, int(extent / tick + 1) * tick)
    label(12, 338, "How much does the frozen bit help its reader?", size=20, bold=True)
    label(12, 317, "Mean paired Brier gain from valid uploads over a matched side-only fit.", size=12, color=mute)
    for i, r in enumerate(rows):
        label(12, 257 - i * 31, r if r in ("C", "D", "MLP") else r.title())
    for col, arm in enumerate(("C", "D", "MLP")):
        center, half = 228 + col * 255, 108
        label(center, 293, f"Frozen {arm} sender", align="middle", bold=True)

        def x(value, center=center, half=half):
            return center + value / limit * half

        for j in range(-round(limit / tick), round(limit / tick) + 1):
            value = j * tick
            line(x(value), 88, x(value), 275, colors.HexColor("#D8E0E3") if j else mute, 0.7 if j else 1.2)
            label(x(value), 73, f"{value:.2f}", size=8, align="middle", color=mute)
        x_oracle = x(data[f"{arm}-to-C"]["oracle_value"])
        for y0 in range(88, 274, 7):
            line(x_oracle, y0, x_oracle, min(y0 + 3, 275), red, 1.2)
        for i, r in enumerate(rows):
            result = data[f"{arm}-to-{r}"]
            y = 261 - 31 * i
            lo, hi = result["paired_seed_panel_bootstrap_95"]
            line(x(lo), y, x(hi), y, blue, 2)
            line(x(lo), y - 2.5, x(lo), y + 2.5, blue, 1.2)
            line(x(hi), y - 2.5, x(hi), y + 2.5, blue, 1.2)
            drawing.add(Circle(x(result["learned_value"]), y, 3.2, strokeColor=blue, fillColor=blue))
    label(12, 43, "Bars show a paired 95% seed/panel bootstrap interval. Positive values favor the upload.", color=mute)
    label(
        12,
        24,
        "Dashed marks show the exact frame-calibrated Bayes gain. Three training seeds, four independent panels.",
        color=mute,
    )
    renderSVG.drawToFile(drawing, str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="optional SVG figure (requires ReportLab)")
    args = parser.parse_args()
    out = args.out
    path = out / "atlas-input.json"
    payload = json.dumps(atlas_rows(out), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing public release differs")
    if not path.exists():
        path.write_text(payload, encoding="utf-8", newline="\n")
    if args.figure:
        plot(read(out / "results.json"), args.figure)
    print(f"{path}: 216 fitted receivers")
