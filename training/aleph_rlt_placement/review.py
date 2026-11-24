"""Audit and rank matched public development receipts. Requires only Python 3.10+.

python training/aleph_rlt_placement/review.py --root runs/placement --out runs/review.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

SEEDS = {137, 163, 191}
CHECKED = ("lean", "rule", "repair", "counter")
METRICS = {"text": "text_nll", "lean": "lean_model", "rule": "rule_joint", "repair": "repair", "counter": "counter"}
CONTROL = {
    "first_penultimate_last": "first_penultimate_damp",
    "global_full": "global_full_damp",
    "core_edges": "core_edges_damp",
    "pass_rlt": "pass_damp",
}
CANDIDATES = ("first_last", "penultimate", *CONTROL)
ALL_ARMS = {"direct", "terminal", *CANDIDATES, *CONTROL.values()}
REGISTERED = {
    2: {"direct", *CANDIDATES, "first_penultimate_damp", "global_full_damp"},
    3: {"direct", "core_edges", "core_edges_damp", "pass_rlt", "pass_damp"},
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_macro(metrics):
    return fmean(metrics[METRICS[lane]] for lane in CHECKED)


def mean_metrics(rows):
    rows = list(rows)
    if any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("unaligned metric names")
    return {k: fmean(row[k] for row in rows) for k in sorted(rows[0])}


def units(rows):
    """Return signed-score ingredients at the sampling unit and verifiable endpoint outcomes."""
    clusters = {lane: defaultdict(lambda: [0.0, 0]) for lane in METRICS}
    outcomes = {lane: {} for lane in CHECKED}
    goals = defaultdict(list)
    episodes = defaultdict(list)
    seen = set()
    for r in rows:
        lane, root, group = r["lane"], r["root"], r["group"]
        key = (lane, root, group, r["order"])
        if key in seen:
            raise ValueError(f"duplicate actor row {key}")
        seen.add(key)
        if lane == "lean":
            goals[root].append(r)
        elif lane == "rule":
            episode, turn = group.rsplit("/", 1)
            episodes[(root, episode)].append((turn, bool(r["hit"])))
        elif lane == "text":
            val, n = r["nll"], r["positions"]
            if not math.isfinite(val) or val < 0 or n <= 0:
                raise ValueError("invalid text loss")
            clusters[lane][group, root] = [val, n]
        else:
            outcomes[lane][root] = bool(r["hit"])
            clusters[lane]["all", root] = [int(r["hit"]), 1]
    for root, candidates in goals.items():
        if len({r["group"] for r in candidates}) != 1:
            raise ValueError("a proof goal has multiple family IDs")
        prefix = sorted(candidates, key=lambda r: (-r["score"], r["order"]))[:4]
        valid = any(r["grade"] == 1 for r in prefix)
        outcomes["lean"][root] = valid
        unit = clusters["lean"]["all", candidates[0]["group"]]
        unit[0] += int(valid)
        unit[1] += 1
    for (root, episode), turns in episodes.items():
        if sorted(t for t, _ in turns) != ["after", "before"]:
            raise ValueError("incomplete two-turn revision")
        valid = all(correct for _, correct in turns)
        outcomes["rule"][episode] = valid
        unit = clusters["rule"]["all", root]
        unit[0] += int(valid)
        unit[1] += 1
    return clusters, outcomes, digest(sorted(seen))


def validate_metrics(metrics, clusters):
    for lane, name in METRICS.items():
        vals = clusters[lane].values()
        if not vals:
            raise ValueError(f"missing {lane} in five-lane study")
        actual = sum(x for x, _ in vals) / sum(n for _, n in vals)
        if not math.isclose(actual, metrics[name], abs_tol=1e-8):
            raise ValueError(f"metric differs from scored rows: {name}")


def audit_cell(path):
    data = load(path)
    cfg, ledger = data["config"], data["ledger"]
    clusters, _, panel = units(data["rows"])
    validate_metrics(data["native"], clusters)
    expected_block_work = (2 + 10 * cfg["r"] + 2 * cfg["s"]) * ledger["padded_positions"]
    if data["train_block_positions"] != expected_block_work or ledger["train_s"] <= 0:
        raise ValueError(f"invalid training ledger in {path}")
    if ledger["lane_examples"] != {k: v * cfg["multiplier"] * cfg["steps"] for k, v in cfg["counts"].items()}:
        raise ValueError(f"actual examples differ from registered mixture in {path}")
    native_key = f"{cfg['arm']}@{cfg['r']}"
    expected = {f"{a}@{cfg['r']}" for a in ALL_ARMS} | {f"{cfg['arm']}@{k}" for k in (1, 2, 3, 4, 6)}
    if set(data["sweep"]) != expected:
        raise ValueError(f"missing runtime or depth control in {path}")
    native_probe = load(path.with_name(f"sweep-{native_key}.json"))
    _, old, panel_hash = units(native_probe["rows"])
    transitions = {}
    for key, values in data["sweep"].items():
        part = native_probe if key == native_key else load(path.with_name(f"sweep-{key}.json"))
        new_clusters, new, identity = units(part["rows"])
        validate_metrics(part["metrics"], new_clusters)
        if part["cfg"] != digest(cfg) or part["metrics"] != values or identity != panel_hash:
            raise ValueError(f"mismatched diagnostic panel {path}: {key}")
        by_lane = {}
        for lane in CHECKED:
            a, b = old[lane], new[lane]
            if a.keys() != b.keys():
                raise ValueError("unpaired sweep outcomes")
            by_lane[lane] = {
                "retained": sum(a[k] and b[k] for k in a),
                "lost": sum(a[k] and not b[k] for k in a),
                "gained": sum(not a[k] and b[k] for k in a),
                "cases": len(a),
            }
        transitions[key] = by_lane
    data.update(clusters=clusters, panel=panel, probe_panel=panel_hash, transitions=transitions)
    return data


def telemetry(cell):
    grad = cell["ledger"]["grad"]
    snapshots = [g["preclip_l2"] for g in grad]
    blocks = [v for g in grad for v in g["blocks_postclip_l2"]]
    if not all(math.isfinite(v) and v >= 0 for v in snapshots + blocks):
        raise ValueError("nonfinite gradient diagnostic")
    traces = {}
    for t in cell["trace"]:
        rows = t["visits"]
        residuals = [row["anchor_delta_rms"] for row in rows]
        errors = [abs(v - row["alpha"] * row["proposed_delta_rms"]) for row, v in zip(rows, residuals, strict=True)]
        if max(errors) > 1e-4 * max(1.0, max(residuals)):
            raise ValueError("realized displacement is inconsistent with the blend clock")
        traces[t["lane"]] = {
            "first_input_rms": rows[0]["input_rms"],
            "final_output_rms": rows[-1]["output_rms"],
            "peak_output_rms": max(r["output_rms"] for r in rows),
            "mean_site_displacement": fmean(r["site_delta_rms"] for r in rows),
            "mean_anchor_displacement": fmean(residuals),
            "peak_blend_error": max(errors),
        }
    return {
        "preclip_l2": snapshots,
        "min_block_gradient": min(blocks),
        "max_block_gradient": max(blocks),
        "zero_block_snapshots": blocks.count(0),
        "probe_trajectories": traces,
    }


def matched(a, b):
    cfg_a = {k: v for k, v in a["config"].items() if k != "arm"}
    cfg_b = {k: v for k, v in b["config"].items() if k != "arm"}
    fields = ("params", "panel", "probe_panel", "train_block_positions")
    counters = ("plan_hash", "input_positions", "padded_positions", "loss_positions", "lane_examples")
    if (
        cfg_a != cfg_b
        or any(a[k] != b[k] for k in fields)
        or any(a["ledger"][k] != b["ledger"][k] for k in counters)
        or a["baselines"] != b["baselines"]
    ):
        raise ValueError("paired arms differ in configuration, evidence, root panel or executed workload")


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    low = int(pos)
    return values[low] + (pos - low) * (values[min(low + 1, len(values) - 1)] - values[low])


def paired_intervals(pairs, draws):
    """Resample shared public clusters within each corpus/lane, with seed effects paired and averaged."""
    by_lane = {}
    series = {}
    for lane in METRICS:
        keys = sorted(pairs[0][0]["clusters"][lane])
        strata = defaultdict(list)
        for stratum, k in keys:
            units_a = [a["clusters"][lane][stratum, k] for a, _ in pairs]
            units_b = [b["clusters"][lane][stratum, k] for _, b in pairs]
            denoms = {n for _, n in units_a + units_b}
            if len(denoms) != 1:
                raise ValueError("unequal paired cluster sizes")
            strata[stratum].append((fmean(v for v, _ in units_a) - fmean(v for v, _ in units_b), denoms.pop()))
        rng = random.Random("aleph-review-v1:" + lane)
        samples = []
        for _ in range(draws):
            total, count = 0.0, 0
            for values in strata.values():
                for num, den in rng.choices(values, k=len(values)):
                    total += num
                    count += den
            samples.append(total / count)
        series[lane] = samples
        by_lane[lane] = {
            "clusters": len(keys),
            "percentile95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        }
    macro = [fmean(tup) for tup in zip(*(series[lane] for lane in CHECKED), strict=True)]
    by_lane["checked_macro"] = {"percentile95": [percentile(macro, 0.025), percentile(macro, 0.975)]}
    return by_lane


def compare(candidate, reference, draws):
    seeds = sorted(candidate.keys() & reference.keys())
    pairs = [(candidate[s], reference[s]) for s in seeds]
    shifts = []
    for a, b in pairs:
        matched(a, b)
        am, bm = a["native"], b["native"]
        shifts.append(
            {
                "nll": am["text_nll"] - bm["text_nll"],
                "text_relative": am["text_nll"] / bm["text_nll"] - 1,
                "checked_macro": checked_macro(am) - checked_macro(bm),
                "train_time_relative": a["ledger"]["train_s"] / b["ledger"]["train_s"] - 1,
                "nonpadding_throughput_relative": b["ledger"]["train_s"] / a["ledger"]["train_s"] - 1,
            }
        )
    return {
        "seeds": seeds,
        "paired_delta": mean_metrics(shifts),
        "per_seed": dict(zip(seeds, shifts, strict=True)),
        "descriptive_intervals": paired_intervals(pairs, draws),
        "meaningful_seed_count": sum(d["nll"] <= -0.02 or d["checked_macro"] >= 0.03 for d in shifts),
    }


def rank(cells, draws):
    by_arm = defaultdict(dict)
    for cell in cells:
        arm, seed = cell["config"]["arm"], cell["config"]["seed"]
        if seed in by_arm[arm]:
            raise ValueError(f"duplicate cell for {arm=} {seed=}; review a single cohort")
        by_arm[arm][seed] = cell
    depth = cells[0]["config"]["r"]
    completed = all(set(by_arm.get(k, {})) >= SEEDS for k in REGISTERED[depth])
    direct = by_arm.get("direct", {})
    competence = {
        k: c["native"]["text_nll"] < c["baselines"]["text_unigram_nll"]
        and any(c["native"][lane] > c["baselines"][lane + "_constant"] for lane in ("repair", "counter"))
        for k, c in direct.items()
    }
    summaries, eligible = {}, []
    for arm, trained in sorted(by_arm.items()):
        m = mean_metrics([c["native"] for c in trained.values()])
        entry = {
            "seeds": sorted(trained),
            "native": m,
            "checked_macro": checked_macro(m),
            "baselines": mean_metrics([c["baselines"] for c in trained.values()]),
            "telemetry_by_seed": {s: telemetry(c) for s, c in trained.items()},
            "mean_train_s": fmean(c["ledger"]["train_s"] for c in trained.values()),
            "mean_nonpadding_positions_per_fit_second": fmean(
                c["ledger"]["input_positions"] / c["ledger"]["train_s"] for c in trained.values()
            ),
            "per_seed_work": {
                s: {
                    **{
                        k: c["ledger"][k]
                        for k in ("input_positions", "padded_positions", "loss_positions", "plan_hash")
                    },
                    "training_block_positions": c["train_block_positions"],
                    "fit_seconds": c["ledger"]["train_s"],
                    "eval_session_seconds": c["eval_session_s"],
                }
                for s, c in trained.items()
            },
            "trainable_params": next(iter(trained.values()))["params"],
        }
        entry["sweeps"] = {}
        for key in sorted(next(iter(trained.values()))["sweep"]):
            entry["sweeps"][key] = {
                "mean": mean_metrics([c["sweep"][key] for c in trained.values()]),
                "paired_episode_seed_transitions": {
                    lane: {
                        field: sum(c["transitions"][key][lane][field] for c in trained.values())
                        for field in ("retained", "lost", "gained", "cases")
                    }
                    for lane in CHECKED
                },
            }
        if arm != "direct" and direct.keys() & trained.keys():
            d = compare(trained, direct, draws)
            entry["versus_direct"] = d
            delta = d["paired_delta"]
            entry["passes_development_thresholds"] = (
                arm in CANDIDATES
                and sum(competence.get(s, False) for s in d["seeds"]) >= 2
                and d["meaningful_seed_count"] >= 2
                and delta["text_relative"] <= 0.01
                and delta["checked_macro"] >= -0.02
            )
            if entry["passes_development_thresholds"] and set(d["seeds"]) >= SEEDS:
                eligible.append(arm)
        control = CONTROL.get(arm)
        if control and trained.keys() & by_arm.get(control, {}).keys():
            entry["matched_damping"] = {"arm": control, **compare(trained, by_arm[control], draws)}
        if arm in ("first_last", "penultimate"):
            entry["coefficient_identity"] = "exact fixed scalar on one nonterminal site"
        if depth == 2 and arm in ("pass_rlt", "core_edges"):
            entry["coefficient_identity"] = "exact matched damping at two passes"
        summaries[arm] = entry
    ordered = []
    while eligible:
        best = min(summaries[a]["native"]["text_nll"] for a in eligible)
        tied = [a for a in eligible if summaries[a]["native"]["text_nll"] <= best + 0.02]
        chosen = min(tied, key=lambda a: (-summaries[a]["checked_macro"], summaries[a]["native"]["text_nll"], a))
        ordered.append(chosen)
        eligible.remove(chosen)
    missing = {a: sorted(SEEDS - set(by_arm.get(a, {}))) for a in sorted(REGISTERED[depth])}
    same_seeds = defaultdict(list)
    for arm, record in summaries.items():
        same_seeds[tuple(record["seeds"])].append(arm)
    text_order = {
        ",".join(map(str, seeds)): sorted(arms, key=lambda a: (summaries[a]["native"]["text_nll"], a))
        for seeds, arms in same_seeds.items()
    }
    return {
        "complete_registered_screen": completed,
        "missing_cells": {a: s for a, s in missing.items() if s},
        "competent_direct_by_seed": competence,
        "ranked_eligible": ordered,
        "descriptive_native_text_order_by_seed_set": text_order,
        "development_recommendation": ("pending" if not completed else ordered[0] if ordered else "hold"),
        "arms": summaries,
    }


def raw_screens(root):
    outputs = {}
    for path in sorted(root.rglob("result.json")):
        data = load(path)
        cfg = data.get("config", {})
        if "eval_n" not in cfg or "dev_native" not in data or not data.get("complete"):
            continue
        key = str(path.parent.parent.relative_to(root))
        arm, seed = cfg["arm"], str(cfg["seed"])
        panel = data["dev_native"]
        preds = Counter(r["prediction"] for cell in panel for r in cell["rows"])
        outputs.setdefault(key, {}).setdefault(arm, {})[seed] = {
            "params": data["params"],
            "r": cfg["r"],
            "steps": data["step"],
            "correct": sum(cell["correct"] for cell in panel),
            "n": sum(cell["n"] for cell in panel),
            "predicted_values": len(preds),
            "modal_prediction_share": max(preds.values()) / preds.total(),
            "by_family_ood": {f"{c['family']}/{c['ood']}": [c["correct"], c["n"]] for c in panel},
        }
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    if args.draws < 100 or not args.root.is_dir():
        parser.error("existing input root and at least 100 bootstrap draws required")
    cohorts = defaultdict(list)
    sources = {}
    for path in sorted(args.root.rglob("receipt.json")):
        data = load(path)
        if data.get("config", {}).get("schema") != 2 or "native" not in data:
            continue
        cell = audit_cell(path)
        cfg = cell["config"]
        key = (
            f"w{cfg['width']}-r{cfg['r']}-u{cfg['steps']}-"
            + digest({k: v for k, v in cfg.items() if k not in ("seed", "arm")})[:12]
        )
        cohorts[key].append(cell)
        sources[path.relative_to(args.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "schema": "aleph-rlt-development-review-v1",
        "receipt_sha256": sources,
        "bootstrap_draws": args.draws,
        "uncertainty": "Paired seed-mean differences on the public development panels. Percentile intervals "
        "resample windows within each text corpus, Lean goal families, revision lineages and finite task roots. "
        "Each sampled cluster is shared across the fixed seeds. Document membership and independently written "
        "generator replicates are unavailable. Intervals are descriptive and unadjusted after arm selection.",
        "raw_valid_label_uniform_accuracy": 1 / 16,
        "raw_screens": raw_screens(args.root),
        "mixed": {k: rank(v, args.draws) for k, v in sorted(cohorts.items())},
    }
    controls_path = args.root / "strong-controls.json"
    if controls_path.is_file():
        controls = load(controls_path)
        result["additional_public_controls"] = {
            "sha256": hashlib.sha256(controls_path.read_bytes()).hexdigest(),
            "view_digest": controls["data"]["view_digest"],
            "native_panel_digest": controls["native_panel_digest"],
            "native": controls["native"],
            "scope": controls["scope"],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.with_suffix(".tmp")
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    out.replace(args.out)
    print(
        json.dumps(
            {
                "mixed_cells": len(sources),
                "cohorts": {k: v["development_recommendation"] for k, v in result["mixed"].items()},
                "raw_cells": sum(len(c) for group in result["raw_screens"].values() for c in group.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
