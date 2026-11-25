"""Audit matched scale receipts and project the strict-1B resource scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "aleph_rlt_placement"
sys.path.insert(0, str(LEGACY))
from corpus import VOCAB, load_corpus  # noqa: E402
from fullscale import CONFIG, block_parameters, work_ledger  # noqa: E402
from study import SEEDS, choose, tensor  # noqa: E402

WIDTHS = (80, 256, 480, 680)
CHECKED = ("lean_model", "rule_joint", "repair", "counter")
ARM_NAMES = {
    "direct": "Direct recurrence",
    "global_full_damp": "Global constant damping",
    "prelude_reread": "Prelude reread",
    "site_damp": "Local reread control",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    body = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == body:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(path)


def stats(values):
    return {"mean": mean(values), "median": median(values), "min": min(values), "max": max(values)}


def geometry(width):
    plain = block_parameters(width, width // 4, 4 * width)
    core = block_parameters(width, width // 4, 35 * width // 8)
    unique = 4 * plain + 10 * core + VOCAB * width + width
    logical = 4 * plain + 20 * core
    return {"params": unique, "logical_block_parameters": logical, "physical_blocks": 14, "visits": 24}


def streams(pack):
    train, _, source = load_corpus(pack)
    train["lean_pos"] = [r for r in train["lean"] if r.good == (1,)]
    train["lean_neg"] = [r for r in train["lean"] if r.good == (0,)]
    counts = {}
    for seed in SEEDS:
        digest = hashlib.sha256()
        tokens = padded = squared = 0
        counts[seed] = {}
        for step in range(1, 33):
            x, lengths = tensor(choose(train, seed, step - 1, 1))
            digest.update(x.numpy().tobytes())
            tokens += int(lengths.sum())
            padded += x.numel()
            squared += x.numel() * x.shape[1]
            if step in (16, 32):
                counts[seed][step] = {
                    "tokens": tokens,
                    "padded": padded,
                    "padded_times_context": squared,
                    "stream": digest.hexdigest(),
                }
    return counts, source


def check_fits(directory, pinned):
    counter, source = pinned
    contract = sha(HERE / "scale_lab.py") + "-" + sha(HERE / "SCALE_PROTOCOL.md")
    rows = []
    for path in sorted(directory.glob("w*-*.json")):
        row = read(path)
        if row["id"] != path.stem or row["contract"] != contract or row["corpus"] != source:
            raise ValueError(f"fit source or identity mismatch in {path.name}")
        if sha(path.with_suffix(".pt")) != row["checkpoint_sha256"]:
            raise ValueError(f"invalid checkpoint bytes: {path.name}")
        d, seed = row["width"], row["seed"]
        if d not in WIDTHS or seed not in SEEDS:
            raise ValueError("unknown rung or seed")
        shape = geometry(d)
        if any(shape[k] != row[k] for k in ("params", "visits", "logical_block_parameters")):
            raise ValueError("independent weight count disagrees")
        if any(counter[seed][32][k] != row[k] for k in ("tokens", "padded", "stream")):
            raise ValueError("corpus replay differs at terminal checkpoint")
        if [x["step"] for x in row["trace"]] != [16, 32]:
            raise ValueError("missing progress interval")
        for snap in row["trace"]:
            c = counter[seed][snap["step"]]
            projection = shape["logical_block_parameters"] - 2 * d * shape["visits"] + VOCAB * d
            flops = 6 * (c["padded"] * projection + 2 * shape["visits"] * c["padded_times_context"] * d)
            if snap["tokens"] != c["tokens"] or snap["train_flops_est"] != flops:
                raise ValueError("token or multiply-add count disagrees on replay")
            snap["padded"] = c["padded"]
            snap["checked_macro"] = mean(snap["metrics"][k] for k in CHECKED)
        rows.append(row)
    return rows, contract


def complete(gates):
    if not gates:
        return False
    highest = max(int(w) for w in gates)
    return highest == WIDTHS[-1] or (highest >= 256 and not gates[str(highest)]["controls_competent"])


def audit_gates(gates, rows):
    arms = ["direct", "global_full_damp", "prelude_reread", "site_damp"]
    output = {}
    previous_failed = False
    for width in WIDTHS:
        items = [r for r in rows if r["width"] == width]
        meta = gates.get(str(width))
        if meta is None:
            status = "not promoted after failed competence gate" if previous_failed else "pending"
            output[str(width)] = {"status": status, "discovered_fits": len(items), **geometry(width)}
            continue
        if previous_failed or meta["trained"] != arms:
            raise ValueError("unauthorized scale promotion")
        cell = {a: sorted((r for r in items if r["arm"] == a), key=lambda r: r["seed"]) for a in arms}
        if any([r["seed"] for r in group] != list(SEEDS) for group in cell.values()):
            raise ValueError("closed rung is incomplete")
        fixed = meta["baselines"]
        finals = {a: [r["trace"][-1] for r in group] for a, group in cell.items()}
        details = {}
        for a, values in finals.items():
            ms = [v["metrics"] for v in values]
            flags = {
                "text_unigram_wins": [
                    s for s, m in zip(SEEDS, ms, strict=True) if m["text_nll"] < fixed["text_unigram_nll"]
                ],
                "lean_or_repair_wins": [
                    s
                    for s, m in zip(SEEDS, ms, strict=True)
                    if m["lean_model"] > m["lean_lexical"] or m["repair"] > 0.25
                ],
                "macro_over_direct": [
                    s
                    for s, x, y in zip(SEEDS, values, finals["direct"], strict=True)
                    if x["checked_macro"] > y["checked_macro"]
                ],
                "mean_text_ratio_vs_direct": mean(m["text_nll"] for m in ms)
                / mean(v["metrics"]["text_nll"] for v in finals["direct"]),
            }
            if "site_damp" in finals and a == "prelude_reread":
                flags["macro_over_site"] = [
                    s
                    for s, p, q in zip(SEEDS, values, finals["site_damp"], strict=True)
                    if p["checked_macro"] > q["checked_macro"]
                ]
            details[a] = flags
        competent = any(
            len(details[a]["text_unigram_wins"]) >= 2 and len(details[a]["lean_or_repair_wins"]) >= 2
            for a in ("direct", "global_full_damp")
        )
        extras = [
            a
            for a in arms[2:]
            if len(details[a]["macro_over_direct"]) >= 2
            and len(details[a].get("macro_over_site", list(SEEDS))) >= 2
            and details[a]["mean_text_ratio_vs_direct"] <= 1.01
        ]
        extras.sort(key=lambda a: -mean(v["checked_macro"] for v in finals[a]))
        arms = ["direct", "global_full_damp", *extras[:1]]
        if competent != meta["controls_competent"] or arms != meta["survivors"]:
            raise ValueError("recomputed promotion disagrees with trainer")
        previous_failed = width >= 256 and not competent
        output[str(width)] = {
            "status": "completed",
            "discovered_fits": len(items),
            **geometry(width),
            "trained": meta["trained"],
            "registered_survivors": arms,
            "baselines": fixed,
            "controls_competent": competent,
            "per_arm_gate_observations": details,
            "next_rung_authorized": width != WIDTHS[-1] and not previous_failed,
        }
    return output


def slopes(rows, rungs):
    """Fit per-seed loss slopes only at matched tokens on completed adjacent rungs."""
    out, series = {}, {}
    for row in rows:
        if rungs[str(row["width"])]["status"] == "completed":
            series[row["arm"], row["width"], row["seed"]] = row
    for low, high in pairwise(WIDTHS):
        arms = sorted({a for a, w, s in series if w == high and all((a, low, t) in series for t in SEEDS)})
        for a in arms:
            for index, step in enumerate((16, 32)):
                raw = {}
                for s in SEEDS:
                    w1, w2 = series[a, low, s], series[a, high, s]
                    h1, h2 = w1["trace"][index], w2["trace"][index]
                    if h1["tokens"] != h2["tokens"] or w1["stream"] != w2["stream"]:
                        raise ValueError("unmatched positions in slope calculation")
                    l1, l2 = (h["metrics"]["text_nll"] for h in (h1, h2))
                    dx = math.log(w2["params"] / w1["params"])
                    raw[str(s)] = {
                        "tokens": h1["tokens"],
                        "text_nll_per_ln_params": (l2 - l1) / dx,
                        "log_text_nll_per_ln_params": math.log(l2 / l1) / dx,
                        "checked_macro_per_ln_params": (h2["checked_macro"] - h1["checked_macro"]) / dx,
                    }
                out[f"{a}-{low}-{high}-step{step}"] = {
                    "seeds": raw,
                    "ranges": {
                        k: stats([v[k] for v in raw.values()]) for k in next(iter(raw.values())) if k != "tokens"
                    },
                }
    return out


def population(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["width"], row["arm"]), []).append(row)
    output = {}
    for (width, arm), group in grouped.items():
        traj = {}
        for i, step in enumerate((16, 32)):
            ss = [r["trace"][i] for r in group]
            traj[str(step)] = {
                "text_nll": stats([t["metrics"]["text_nll"] for t in ss]),
                "checked_macro": stats([t["checked_macro"] for t in ss]),
                "nonpadding_positions": stats([t["tokens"] for t in ss]),
                "padded_positions": stats([t["padded"] for t in ss]),
                "estimated_flops": stats([t["train_flops_est"] for t in ss]),
                "preclip_gradient_norm": stats([t["preclip_norm"] for t in ss]),
            }
        output[f"w{width}-{arm}"] = {
            "count": len(group),
            "seeds": sorted(r["seed"] for r in group),
            "traces": traj,
            **geometry(width),
            "train_seconds": stats([r["train_seconds"] for r in group]),
            "nonpadding_positions_per_s": stats([r["tokens_per_second"] for r in group]),
            "sampled_rss_mib": stats([r["peak_sampled_rss"] / 1024**2 for r in group]),
            "analytical_gflop_per_s": stats([r["flops"] / r["train_seconds"] / 1e9 for r in group]),
        }
    return output


def ledger(rows, rungs):
    cfg = read(CONFIG)
    two, three = (work_ledger(cfg, d) for d in (2, 3))
    if two["unique"] != 999352320 or three["unique"] != two["unique"]:
        raise ValueError("strict dense architecture count changed")
    user = cfg["user_cost_case"]
    rate = user["tokens_per_leg"] / (3600 * user["hours_per_leg"])
    cost_per_token = user["credits_per_leg"] / user["tokens_per_leg"]
    total, reserve = user["credits_available"], 100.0
    spendable = total - reserve
    finished = [int(w) for w, gate in rungs.items() if gate["status"] == "completed"]
    largest = max(finished)
    control = [r for r in rows if r["width"] == largest and r["arm"] == "direct"]
    observed = [r["flops"] / r["train_seconds"] for r in control]
    local = {
        "calibration_width": largest,
        "seed_rate_gflop_s": {str(r["seed"]): r["flops"] / r["train_seconds"] / 1e9 for r in control},
    }
    ratio = three["estimated_training_flops_per_10B"] / two["estimated_training_flops_per_10B"]
    depths = {}
    for d, geom in ((2, two), (3, three)):
        per_token = sum(
            geom[k]
            for k in ("linear_training_flops_per_position", "ideal_causal_attention_training_flops_per_position")
        )
        rel = ratio if d == 3 else 1.0
        targets = {}
        for n in (10**8, 10**9, 10**10, 2 * 10**10):
            f = n * per_token
            years = [f / q / (86400 * 365.25) for q in observed]
            targets[str(n)] = {
                "positions": n,
                "estimated_training_flops": f,
                "one_core_equivalent_years": {"median": median(years), "min": min(years), "max": max(years)},
                "assumed_tpu_hours_if_quote_at_this_depth": n / rate / 3600,
                "assumed_tpu_credits_if_quote_at_this_depth": n * cost_per_token,
                "assumed_tpu_hours_if_quote_at_two_passes": n / rate / 3600 * rel,
                "assumed_tpu_credits_if_quote_at_two_passes": n * cost_per_token * rel,
            }
        depths[str(d)] = {**geom, "targets": targets}
    # Each stage is cumulative and all interim checkpoints are prefixes of the two matched legs.
    stages = []
    for n, trigger in (
        (10**7, "exact restart, identity, loss and throughput instrumentation"),
        (10**8, "billed end-to-end throughput including evaluation and I/O"),
        (5 * 10**8, "verify numeric stability and independent task/checker competence"),
        (10**9, "predeclared futility comparison on independent families"),
        (10**10, "matched 10B checkpoint adjudication"),
    ):
        stages.append(
            {
                "per_arm_positions": n,
                "total_positions": 2 * n,
                "cumulative_hours": 2 * n / rate / 3600,
                "cumulative_credits": 2 * n * cost_per_token,
                "checkpoint_purpose": trigger,
            }
        )
    paid = 2 * 10**10 * cost_per_token
    extension = math.floor((spendable - paid) / cost_per_token / 10**8) * 10**8
    final_credits = paid + extension * cost_per_token
    return {
        "strict_configuration_sha256": sha(CONFIG),
        "memory": {
            "bf16_weight_bytes": two["bf16_weights_bytes"],
            "illustrative_optimizer_inventory_16byte_per_param": two["illustrative_16byte_adam_inventory"],
            "one_bf16_full_context_residual_bytes": 2
            * cfg["architecture"]["max_context"]
            * cfg["architecture"]["width"],
            "one_bf16_full_context_logits_bytes": 2 * cfg["architecture"]["max_context"] * cfg["architecture"]["vocab"],
            "one_core_study_limit_mib": 3072,
        },
        "local": {
            **local,
            "scope": "FLOP-equivalent one-core arithmetic using the three measured terminal direct-loop "
            "rates at the largest completed proxy width. Strict-1B Adam inventory exceeds the study cap. "
            "Proxy counts padded full-square attention; strict counts ideal triangular full-context work. "
            "Training timers exclude evaluation and checkpoint I/O.",
        },
        "depths": depths,
        "assumed_credit_rate_per_hour": user["credits_per_leg"] / user["hours_per_leg"],
        "assumed_nonpadding_positions_per_s": rate,
        "quote_depth_observed": user["calibrated_depth"],
        "spend_status": "HOLD; no paid work authorized",
        "credits_available": total,
        "chosen_reserve_credits": reserve,
        "working_cap_credits": spendable,
        "two_pass_stages_if_operator_rate_applies": stages,
        "paired_10b_then_affordable_extension": {
            "winner_extra_positions": extension,
            "winner_total_positions": 10**10 + extension,
            "all_arms_positions": 2 * 10**10 + extension,
            "total_credits": final_credits,
            "credits_remaining": total - final_credits,
            "total_hours": (2 * 10**10 + extension) / rate / 3600,
        },
        "early_stop_challenger_at_1b_complete_winner_to_20b": {
            "winner_total_positions": 2 * 10**10,
            "challenger_total_positions": 10**9,
            "all_arms_positions": 21 * 10**9,
            "total_credits": 21 * 10**9 * cost_per_token,
            "credits_remaining": total - 21 * 10**9 * cost_per_token,
            "total_hours": 21 * 10**9 / rate / 3600,
            "decision": "Requires independent early futility. The challenger stops before the two-arm 10B comparison.",
        },
        "three_complete_10b_legs_quoted_credits": 3 * 10**10 * cost_per_token,
        "three_leg_over_balance": 3 * 10**10 * cost_per_token - total,
        "three_leg_over_working_cap": 3 * 10**10 * cost_per_token - spendable,
        "three_pass_work_ratio": ratio,
        "three_pass_two_10b_credits_if_quote_at_two_passes": paid * ratio,
        "three_pass_max_equal_arm_positions_under_working_cap_if_quote_at_two_passes": math.floor(
            spendable / (2 * cost_per_token * ratio)
        ),
        "stopping_rule": "Before extending, cap additional nonpadding positions at floor((balance - reserve - spent) / "
        "observed inclusive credits per position). Stop if that quantity cannot fund the next matched stage.",
    }


def atlas_rows(rows, rungs):
    output = []
    for row in rows:
        width, arm = row["width"], row["arm"]
        gate = rungs[str(width)]
        fail = ["32-update public development prefix; eight roots per source group; one CPU core."]
        if gate.get("controls_competent") is False:
            fail.append("Both-control competence gate failed at this rung.")
        if gate.get("trained") and arm not in gate["registered_survivors"]:
            fail.append("Arm was eliminated at this rung.")
        final = row["trace"][-1]
        output.append(
            {
                "key": row["id"],
                "checkpoint": row["id"] + ".pt",
                "sha256": row["checkpoint_sha256"],
                "architecture": f"Dense GQA/SwiGLU shared-core Aleph width {width}, {arm}",
                "params": row["params"],
                "passes": 2,
                "seed": row["seed"],
                "steps": row["steps"],
                "metrics": {
                    **final["metrics"],
                    "checked_macro": final["checked_macro"],
                    "nonpadding_positions": row["tokens"],
                    "padded_positions": row["padded"],
                    "analytic_training_flops": row["flops"],
                    "train_seconds": row["train_seconds"],
                    "tokens_per_second": row["tokens_per_second"],
                    "sampled_peak_rss_mib": row["peak_sampled_rss"] / 1024**2,
                },
                "mechanism": {
                    "logical_blocks": row["visits"],
                    "physical_blocks": 14,
                    "logical_block_parameter_touches": row["logical_block_parameters"],
                    "preclip_grad_norm": final["preclip_norm"],
                    "earlier_step16": row["trace"][0],
                    "ordered_stream_sha256": row["stream"],
                    "corpus": row["corpus"],
                    "recipe": "32 AdamW steps at .0007 with 3 text, 4 Lean, 1 revision, 1 repair, 1 counter per batch",
                },
                "failures": " ".join(fail),
            }
        )
    return output


def figure(summary, dest):
    from reportlab.graphics import renderSVG
    from reportlab.graphics.shapes import Circle, Drawing, Line, String
    from reportlab.lib.colors import HexColor

    colors = {"direct": "#485472", "global_full_damp": "#167A76", "prelude_reread": "#AD5427", "site_damp": "#67508E"}
    draw = Drawing(840, 350)
    ink = HexColor("#1A3040")

    def label(x, y, text, size=11, color=ink, align="start", bold=False):
        draw.add(
            String(
                x,
                y,
                text,
                fontSize=size,
                fontName="Helvetica-Bold" if bold else "Helvetica",
                fillColor=color,
                textAnchor=align,
            )
        )

    def line(x1, y1, x2, y2, color, width=1):
        draw.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))

    label(15, 324, "What changed after 32 matched updates?", 19, bold=True)
    label(15, 302, "Means across three seeds. Bars span the individual seed range on the same public evaluation.")
    admitted = [w for w in WIDTHS if summary["rungs"][str(w)]["status"] == "completed"]
    log_x = [math.log(geometry(w)["params"]) for w in admitted]
    x_range = log_x[-1] - log_x[0] if len(log_x) > 1 else 1.0
    scope = summary["groups"]
    for col, (kind, title) in enumerate(
        (("text_nll", "Text loss (nats per byte)"), ("checked_macro", "Checked macro accuracy"))
    ):
        left, right = 65 + 414 * col, 355 + 414 * col
        label((left + right) / 2, 275, title, 13, align="middle", bold=True)
        yy = [g["traces"]["32"][kind][k] for g in scope.values() if g["count"] == 3 for k in ("min", "max")]
        baseline = summary["rungs"][str(admitted[0])]["baselines"]["text_unigram_nll"] if kind == "text_nll" else None
        if baseline is not None:
            yy.append(baseline)
        lo, hi = min(yy), max(yy)
        pad = max(0.035, (hi - lo) * 0.11)
        lo, hi = max(0, lo - pad), hi + pad

        def ypos(value, lo=lo, hi=hi):
            return 88 + 158 * (value - lo) / (hi - lo)

        for j in range(5):
            value = lo + (hi - lo) * j / 4
            y = ypos(value)
            line(left - 8, y, right + 8, y, HexColor("#D5E0E2"), 0.6)
            label(left - 12, y - 3, f"{value:.2f}", 8, align="end")
        if baseline is not None:
            y = ypos(baseline)
            for dash in range(left - 8, right + 8, 7):
                line(dash, y, min(dash + 3, right + 8), y, ink, 1.0)
        for j, width in enumerate(admitted):
            xx = left + (right - left) * (log_x[j] - log_x[0]) / x_range if len(admitted) > 1 else (left + right) / 2
            label(xx, 72, f"{geometry(width)['params'] / 1e6:.2f}M", 9, align="middle")
        for offset, (arm, color) in enumerate(colors.items()):
            last = None
            for j, width in enumerate(admitted):
                group = scope.get(f"w{width}-{arm}")
                if group is None or group["count"] != 3:
                    continue
                q = group["traces"]["32"][kind]
                xx = (
                    left + (right - left) * (log_x[j] - log_x[0]) / x_range if len(admitted) > 1 else (left + right) / 2
                )
                xx += (offset - 1.5) * 4
                color_code = HexColor(color)
                line(xx, ypos(q["min"]), xx, ypos(q["max"]), color_code)
                if last is not None:
                    line(*last, xx, ypos(q["mean"]), color_code)
                draw.add(Circle(xx, ypos(q["mean"]), 3, fillColor=color_code, strokeColor=color_code))
                last = (xx, ypos(q["mean"]))
    for i, (arm, color) in enumerate(colors.items()):
        xx = 30 + 178 * i
        line(xx, 45, xx + 14, 45, HexColor(color), 2.2)
        label(xx + 18, 41, ARM_NAMES[arm], 10)
    label(
        15, 19, "Horizontal labels give unique model parameters. The dashed line marks the training-byte unigram.", 10
    )
    renderSVG.drawToFile(draw, str(dest))


def markdown(summary):
    rungs, groups = summary["rungs"], summary["groups"]
    largest = max(int(w) for w, r in rungs.items() if r["status"] == "completed")
    widths = [w for w in WIDTHS if rungs[str(w)]["status"] == "completed"]
    count = summary["evaluated_fits"]
    fixed = rungs[str(widths[0])]["baselines"]
    anchor = next(iter(summary["individual_fits"].values()))["trace"][-1]["metrics"]
    lines = [
        "# ALEPH-06 matched scale frontier",
        "",
        f"{count} fits completed on the public mixed-corpus development panel. The largest completed model had "
        f"{geometry(largest)['params']:,} unique parameters. The run measured two recurrent passes, "
        "24 logical block visits and 32 matched optimizer updates. The paid gate remains on hold. "
        "[Registered protocol](SCALE_PROTOCOL.md)",
        "",
        "The scale screen retains direct recurrence and global constant damping as calibration arms. Two-pass "
        "whole-pass RLT has the same two coefficients as its matched pass-damping control. At three passes the "
        "earlier 14M replication favored matched damping on text in all three seeds.",
        "",
        "## Promotion outcomes",
        "",
    ]
    for w in WIDTHS:
        r = rungs[str(w)]
        if r["status"] != "completed":
            lines.append(f"- Width {w}, {r['params']:,} unique parameters, {r['status']}.")
            continue
        controlled = "passed" if r["controls_competent"] else "failed"
        survivors = ", ".join(ARM_NAMES[a].lower() for a in r["registered_survivors"])
        if r["next_rung_authorized"]:
            action = f"The next rung admitted {survivors}."
        elif w == WIDTHS[-1]:
            action = "This was the last registered rung."
        else:
            action = "The larger rungs were withheld."
        lines.append(
            f"- Width {w}, {r['params']:,} unique parameters. The control competence gate {controlled}. {action}"
        )
    lines += [
        "",
        "## Development measurements",
        "",
        f"The training-byte unigram scored {fixed['text_unigram_nll']:.5f} nats and lexical Lean succeeded on "
        f"{anchor['lean_lexical']:.1%} of goals. Transparent source cues reached {fixed['repair_cue']:.1%} on "
        f"repair and {fixed['counter_cue']:.1%} on counterexamples. The registered promotion rule uses repair "
        "chance of 25% or the lexical Lean comparator as its checked threshold.",
        "",
        "Each pair of text or verified-macro numbers gives the seed mean at 16 and 32 updates. Verified macro equally "
        "weights Lean top-four, joint rule revision, repair and counterexamples. Throughput counts nonpadding "
        "positions during optimization. RSS is the largest process sample at an update boundary.",
        "",
        "| Width | Arm | Text NLL, 16 / 32 | Checked macro, 16 / 32 | Median positions/s | Peak RSS, MiB |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for w in widths:
        for arm in rungs[str(w)]["trained"]:
            g = groups[f"w{w}-{arm}"]
            a, b = g["traces"]["16"], g["traces"]["32"]
            lines.append(
                f"| {w} | {ARM_NAMES[arm]} | {a['text_nll']['mean']:.4f} / {b['text_nll']['mean']:.4f} | "
                f"{a['checked_macro']['mean']:.4f} / {b['checked_macro']['mean']:.4f} | "
                f"{g['nonpadding_positions_per_s']['median']:.1f} | {g['sampled_rss_mib']['max']:.1f} |"
            )
    lines += [
        "",
        "All arms and widths replayed their ordered training-position digests. Padding counts and dense "
        "projection, output head and full-square attention FLOPs were reconstructed independently at both "
        "checkpoints. Within each width, parameter count and logical block applications match across arms.",
        "",
        "| Width | Logical block parameters visited per position | Nonpadding positions at 32, seed range | "
        "Padded positions at 32, seed range | Training FLOPs, seed range |",
        "|---:|---:|---:|---:|---:|",
    ]
    for w in widths:
        r = rungs[str(w)]
        t = groups[f"w{w}-direct"]["traces"]["32"]

        def extent(k, t=t):
            return f"{t[k]['min']:,} to {t[k]['max']:,}"

        lines.append(
            f"| {w} | {r['logical_block_parameters']:,} | {extent('nonpadding_positions')} | "
            f"{extent('padded_positions')} | {t['estimated_flops']['min']:.3e} to {t['estimated_flops']['max']:.3e} |"
        )
    lines += [
        "",
        "## Seed-level scale slopes",
        "",
        "The entries are changes in mean held text NLL divided by the natural-log change in unique parameters, "
        "at the same 32-update token prefix and physical recurrence. Negative slopes mean lower text loss at the "
        "larger width. A wider model spends more FLOPs on that prefix. In a law $L(P)=L_\\infty+A P^{-b}$, each "
        "admissible choice of $L_\\infty$ gives a different exponent through two points. These short-prefix "
        "secants cannot identify a 1B winner or an optimum at fixed compute.",
        "",
        "| Widths | Arm | Seed 137 | Seed 163 | Seed 191 | Range |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, val in sorted(summary["matched_log_parameter_slopes"].items()):
        if not key.endswith("step32"):
            continue
        a, lo, hi, _ = key.rsplit("-", 3)
        s = [val["seeds"][str(seed)]["text_nll_per_ln_params"] for seed in SEEDS]
        lines.append(
            f"| {lo} → {hi} | {ARM_NAMES[a]} | {s[0]:+.4f} | {s[1]:+.4f} | {s[2]:+.4f} | "
            f"[{min(s):+.4f}, {max(s):+.4f}] |"
        )
    lines += [
        "",
        "## Strict-1B work equivalent",
        "",
        f"The largest direct proxy used for this comparison had width {largest}. Dividing full-context strict-1B "
        "work by its three measured one-core arithmetic rates gives the following median and range of equivalent "
        "years. The strict model's ~15.99 GB optimizer inventory exceeds the 3 GiB experimental cap. "
        "The full-scale formula assumes ideal triangular attention whereas the proxy counts dense scores over "
        "padded short batches. [Credit stages and memory assumptions](STRICT_1B_BUDGET.md)",
        "",
        "| Nonpadding positions | Two-pass one-core equivalent years | Three-pass one-core equivalent years |",
        "|---:|---:|---:|",
    ]
    for n in (10**8, 10**9, 10**10, 2 * 10**10):
        y = [summary["strict_1b"]["depths"][str(d)]["targets"][str(n)]["one_core_equivalent_years"] for d in (2, 3)]
        cell = [f"{v['median']:.2f} [{v['min']:.2f}, {v['max']:.2f}]" for v in y]
        lines.append(f"| {n:,} | {cell[0]} | {cell[1]} |")
    lines += [
        "",
        "The source corpus is development evidence. Each worker used one allowed logical CPU while other experiments "
        "were active on the host. Shared physical resources can affect timing. Evaluation uses eight roots per source "
        "group and shares them across all arms. The raw "
        "seed metrics, gradients, archive and stream hashes, throughput and exact promotion checks are in the local "
        "machine-readable release.",
        "",
        "## Reproduction",
        "",
        "```text",
        "python training/aleph_next/bounded.py --wall 21600 --core 4 scale_lab.py "
        "--out ../aleph-next/scale --pack runs/placement/mixed",
        "python training/aleph_next/scale_release.py --out ../aleph-next/scale --pack runs/placement/mixed "
        "--figure ../aleph-next/scale/scale-frontier.svg --markdown ../aleph-next/scale/SCALE_RESULTS.md",
        "python -m pytest -q training/aleph_next/test_scale_release.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def release(directory, pack, partial=False):
    gates = read(directory / "gates.json")
    finished = complete(gates)
    if not finished and not partial:
        raise ValueError("a registered rung is still running; --partial writes an explicitly temporary report")
    rows, contract = check_fits(directory, streams(pack))
    rungs = audit_gates(gates, rows)
    models = atlas_rows(rows, rungs)
    summary = {
        "schema": "datagoblin.aleph-scale-release.v1",
        "status": "complete" if finished else "in progress",
        "contract": contract,
        "evaluated_fits": len(rows),
        "rungs": rungs,
        "groups": population(rows),
        "matched_log_parameter_slopes": slopes(rows, rungs),
        "strict_1b": ledger(rows, rungs),
        "individual_fits": {
            r["id"]: {
                k: r[k]
                for k in (
                    "seed",
                    "arm",
                    "width",
                    "params",
                    "logical_block_parameters",
                    "visits",
                    "tokens",
                    "padded",
                    "flops",
                    "train_seconds",
                    "tokens_per_second",
                    "peak_sampled_rss",
                    "checkpoint_sha256",
                    "trace",
                    "stream",
                )
            }
            for r in rows
        },
    }
    suffix = "-preview" if partial else ""
    write(directory / ("summary" + suffix + ".json"), summary)
    write(directory / ("atlas-input" + suffix + ".json"), models)
    print(f"{summary['status']}: {len(rows)} fits across {[w for w, s in rungs.items() if s['status'] == 'completed']}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    p.add_argument("--figure", type=Path, help="optional SVG figure of completed 32-step cohorts")
    p.add_argument("--markdown", type=Path, help="optional standalone report for a completed tournament")
    args = p.parse_args()
    result = release(args.out, args.pack, args.partial)
    if args.figure:
        figure(result, args.figure)
    if args.markdown:
        if result["status"] != "complete":
            raise ValueError("the research report requires a closed tournament")
        args.markdown.write_text(markdown(result), encoding="utf-8", newline="\n")
