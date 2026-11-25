"""Paired document, program-orbit and graph-lineage statistics for frozen 14M terminals."""

from collections import defaultdict

import numpy as np

ARMS = ("direct", "global_full_damp")
SEEDS = (137, 163, 191)
MODEL_NAMES = tuple(f"{a}-{s}" for a in ARMS for s in SEEDS)
REASON_TASKS = ("repair", "revision")
BOOT_SEED = 202611253
DRAWS = 20000


def band(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def cube(all_scores, task):
    base = all_scores[MODEL_NAMES[0]]
    selection = [u for u in base["clusters"] if u["task"] == task]

    def ids(row):
        return row["family"], row["template"], row["root"]

    ordered = sorted(selection, key=ids)
    by_seed = []
    for name in MODEL_NAMES:
        units = sorted((u for u in all_scores[name]["clusters"] if u["task"] == task), key=ids)
        if [ids(u) for u in units] != [ids(u) for u in ordered] or any(
            {k: v for k, v in u["scores"].items() if k != "model"}
            != {k: v for k, v in v0["scores"].items() if k != "model"}
            for u, v0 in zip(units, ordered, strict=True)
        ):
            raise ValueError("unmatched unit identities or cue outcomes")
        by_seed.append([u["scores"]["model"] for u in units])
    families = sorted({u["family"] for u in ordered})
    templates = defaultdict(set)
    counts = defaultdict(int)
    for unit in ordered:
        templates[unit["family"]].add(unit["template"])
        counts[unit["family"], unit["template"]] += 1
    ts = {len(v) for v in templates.values()}
    if len(ts) != 1 or len(set(counts.values())) != 1 or len({ids(u) for u in ordered}) != len(ordered):
        raise ValueError("unbalanced or duplicated generator units")
    cues = sorted(k for k in ordered[0]["scores"] if k != "model")
    cols = [*by_seed, *[[u["scores"][k] for u in ordered] for k in cues]]
    mat = np.asarray(cols, dtype=float).reshape(len(cols), len(families), ts.pop(), next(iter(counts.values())))
    if np.any((mat < 0) | (mat > 1)) or not np.isfinite(mat).all():
        raise ValueError("invalid unit utility")
    return mat, cues, families


def reasoning_draw(grid, count, rng, resample_families):
    _, families, templates, roots = grid.shape
    fi = (
        rng.integers(families, size=(count, families))
        if resample_families
        else np.broadcast_to(np.arange(families), (count, families))
    )
    ri = rng.integers(roots, size=(count, families, templates, roots))
    return grid[:, fi[:, :, None, None], np.arange(templates)[None, None, :, None], ri].mean(axis=(2, 3, 4)).T


def text_draw(grid, labels, count, rng):
    idx = []
    for stratum in sorted(set(labels)):
        block = np.flatnonzero(labels == stratum)
        idx.append(rng.choice(block, (count, len(block)), replace=True))
    return grid[:, np.concatenate(idx, axis=1)].mean(axis=2).T


def reduce_seed(draw, idx):
    models = draw[:, :6].reshape(-1, 2, 3)
    return np.take_along_axis(models, idx[:, None, :], axis=2).mean(axis=2)


def bootstrap(grids, tgrid, strata, draws=DRAWS):
    """Common resampling in both architectures. Independent tasks keep their own data clusters."""
    collection = defaultdict(lambda: defaultdict(list))
    rng = np.random.default_rng(BOOT_SEED)
    for lo in range(0, draws, 128):
        count = min(128, draws - lo)
        regular = [reasoning_draw(g, count, rng, True) for g in grids]
        fixed = [reasoning_draw(g, count, rng, False) for g in grids]
        t = text_draw(tgrid, strata, count, rng)
        flat = np.broadcast_to(np.arange(3), (count, 3))
        sample = rng.integers(3, size=(count, 3))
        for mode, r, idx in (
            ("primary", regular, flat),
            ("crossed_seeds", regular, sample),
            ("fixed_families", fixed, flat),
        ):
            by_task = [reduce_seed(x, idx) for x in r]
            macro = (by_task[0] + by_task[1]) / 2
            nll = reduce_seed(t, idx)
            store = collection[mode]
            store["checked_gain"].append(macro[:, 1] - macro[:, 0])
            store["relative_text_gain"].append(1 - nll[:, 1] / nll[:, 0])
            for a, arm in enumerate(ARMS):
                store[arm + "/unigram_gain_nats"].append(t[:, 6] - nll[:, a])
                for task, score, ensemble in zip(REASON_TASKS, by_task, r, strict=True):
                    store[arm + "/" + task + "_cue_gain"].append(score[:, a] - ensemble[:, 6:].max(axis=1))
    return {mode: {k: band(np.concatenate(v)) for k, v in group.items()} for mode, group in collection.items()}


def summarize_task(grid, cues, families):
    means = grid.mean(axis=(1, 2, 3))
    table = {name: float(x) for name, x in zip(cues, means[6:], strict=True)}
    family = {}
    for i, f in enumerate(families):
        values = grid[:, i].mean(axis=(1, 2))
        family[f] = {a: float(x) for a, x in zip(ARMS, values[:6].reshape(2, 3).mean(1), strict=True)}
        family[f]["strongest_cue"] = float(values[6:].max())
        family[f]["strongest_cue_names"] = [c for c, v in zip(cues, values[6:], strict=True) if v == values[6:].max()]
        family[f]["simple_cue_hits_ceiling"] = bool(values[6:].max() == 1)
        family[f]["paired_seed_gains"] = {
            str(s): float(g) for s, g in zip(SEEDS, values[3:6] - values[:3], strict=True)
        }
        family[f]["b_minus_a"] = family[f][ARMS[1]] - family[f][ARMS[0]]
    return {
        "families": family,
        "all_cues": table,
        "strongest_cue": max(table, key=table.get),
        "strongest_cue_accuracy": max(table.values()),
        "exact_verifier": 1.0,
        "family_count": len(families),
        "independent_roots": int(np.prod(grid.shape[1:])),
        "auxiliary_arms": {},
        "models": {
            a: {"mean": float(x.mean()), "per_seed": dict(zip(map(str, SEEDS), map(float, x), strict=True))}
            for a, x in zip(ARMS, means[:6].reshape(2, 3), strict=True)
        },
    }


def compare(scores, doc_strata, text_baselines, draws=DRAWS):
    if set(scores) != set(MODEL_NAMES):
        raise ValueError("six frozen terminal records required")
    grids = []
    tasks = {}
    for task in (*REASON_TASKS, "archived_revision"):
        grid, cues, families = cube(scores, task)
        tasks[task] = summarize_task(grid, cues, families)
        if task in REASON_TASKS:
            grids.append(grid)
        for a in ARMS:
            per_seed = {}
            for s in SEEDS:
                cluster = [u for u in scores[f"{a}-{s}"]["clusters"] if u["task"] == task]
                per_seed[str(s)] = {
                    k: float(np.mean([x["diagnostics"][k] for x in cluster])) for k in cluster[0]["diagnostics"]
                }
            tasks[task]["auxiliary_arms"][a] = per_seed
    docs = sorted(doc_strata)
    if len(docs) != 12 or any(set(v["text"]["per_document"]) != set(docs) for v in scores.values()):
        raise ValueError("twelve identical authored document clusters required")
    if len({v["text"]["panel_sha256"] for v in scores.values()}) != 1:
        raise ValueError("different authored text draws")
    grid = np.asarray(
        [[scores[name]["text"]["per_document"][d] for d in docs] for name in MODEL_NAMES]
        + [
            [text_baselines["unigram"]["per_document"][d] for d in docs],
            [text_baselines["interpolated_trigram"]["per_document"][d] for d in docs],
        ]
    )
    if not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError("invalid document-level probability loss")
    intervals = bootstrap(grids, grid, np.asarray([doc_strata[d] for d in docs]), draws)
    primary = intervals["primary"]
    nll = grid[:6].mean(axis=1).reshape(2, 3)
    checked = np.mean([r[:6].mean(axis=(1, 2, 3)).reshape(2, 3) for r in grids], axis=0)
    text_gain = float(1 - nll[1].mean() / nll[0].mean())
    checked_gain = float(checked[1].mean() - checked[0].mean())
    seed_text = 1 - nll[1] / nll[0]
    seed_checked = checked[1] - checked[0]
    competence = {}
    for i, a in enumerate(ARMS):
        better = nll[i] < grid[6].mean()
        text_ok = bool(nll[i].mean() < grid[6].mean() and sum(better) >= 2)
        details = {}
        for task in REASON_TASKS:
            lane = tasks[task]
            above = [s for s in SEEDS if lane["models"][a]["per_seed"][str(s)] > lane["strongest_cue_accuracy"]]
            diff = lane["models"][a]["mean"] - lane["strongest_cue_accuracy"]
            interval = primary[a + "/" + task + "_cue_gain"]
            details[task] = {
                "advantage": diff,
                "interval": interval,
                "positive_seeds": above,
                "pass": len(above) >= 2 and diff > 0 and interval[0] > 0,
            }
        competence[a] = {
            "mean_text_nll": float(nll[i].mean()),
            "unigram": float(grid[6].mean()),
            "trigram_nll_minus_model_nll": float(grid[7].mean() - nll[i].mean()),
            "trigram_win_seeds": [s for s, x in zip(SEEDS, nll[i], strict=True) if x < grid[7].mean()],
            "text_unigram_pass": text_ok,
            "unigram_win_seeds": [s for s, ok in zip(SEEDS, better, strict=True) if ok],
            "checked_task_checks": details,
            "qualifying_checked_tasks": sum(x["pass"] for x in details.values()),
            "pass": text_ok and any(x["pass"] for x in details.values()),
        }
    textual = bool(text_gain >= 0.01 and primary["relative_text_gain"][0] > 0 and (seed_text > 0).sum() >= 2)
    reasoning = bool(checked_gain >= 0.02 and primary["checked_gain"][0] > 0 and (seed_checked > 0).sum() >= 2)
    safe = {
        "text_within_1pct": primary["relative_text_gain"][0] >= -0.01,
        "checked_within_2pp": primary["checked_gain"][0] >= -0.02,
    }
    sensitivity = {}
    for name in REASON_TASKS:
        other = next(t for t in REASON_TASKS if t != name)
        table = tasks[name]["families"]
        other_gain = tasks[other]["models"][ARMS[1]]["mean"] - tasks[other]["models"][ARMS[0]]["mean"]
        for family in table:
            val = np.mean([r["b_minus_a"] for f, r in table.items() if f != family])
            sensitivity[name + "/" + family] = float((other_gain + val) / 2)
    return {
        "schema": "datagoblin.14m-fresh-gated-comparison.v1",
        "tasks": tasks,
        "held_competence": competence,
        "held_gates_passed": bool(
            all(x["pass"] for x in competence.values()) and all(safe.values()) and (textual or reasoning)
        ),
        "safeguards": safe,
        "benefit_routes": {"text": textual, "checked": reasoning},
        "observed": {
            "relative_text_gain": text_gain,
            "checked_gain": checked_gain,
            "checked_macro_by_arm": dict(zip(ARMS, map(float, checked.mean(axis=1)), strict=True)),
            "relative_text_gain_by_seed": dict(zip(map(str, SEEDS), map(float, seed_text), strict=True)),
            "checked_gain_by_seed": dict(zip(map(str, SEEDS), map(float, seed_checked), strict=True)),
            "text_nll_by_arm_seed": {
                a: dict(zip(map(str, SEEDS), map(float, x), strict=True)) for a, x in zip(ARMS, nll, strict=True)
            },
            "checked_macro_by_arm_seed": {
                a: dict(zip(map(str, SEEDS), map(float, x), strict=True)) for a, x in zip(ARMS, checked, strict=True)
            },
        },
        "intervals": intervals,
        "leave_one_reason_family_out_checked_gain": sensitivity,
        "text_controls": {
            "unigram_nats_per_byte": float(grid[6].mean()),
            "interpolated_trigram_nats_per_byte": float(grid[7].mean()),
        },
        "text_doc_gains_B_over_A_nats": {
            d: float(grid[:3, i].mean() - grid[3:6, i].mean()) for i, d in enumerate(docs)
        },
        "bootstrap": {
            "draws": draws,
            "statistics_seed": BOOT_SEED,
            "primary": "fixed observed training seeds with paired stratified authors, sampled generator classes "
            "and intact program/graph roots",
            "interpretation": "Four repair and two graph regimes limit external validity.",
        },
    }
