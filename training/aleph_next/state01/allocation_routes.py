"""Fit the external policies on measured train actions, freeze, then replay held policies."""

import argparse
import itertools
import os
import time
from array import array
from pathlib import Path

import numpy as np
import torch
from allocation_replay import DEFAULT, HERE, MEDIAN, file_hash, pin, read, save
from allocation_tasks import ACTIONS, SEEDS, digest
from torch import nn

from datagoblin.experiments.ldt_array_sorter import make_model

PRIMARY = 4 * MEDIAN
ORDERS = tuple(itertools.permutations(ACTIONS))
REFERENCE = ("congruence", "normal", "frontiers", "feedback")
WEIGHTS = {"architecture": {"family": "latent_array", "hidden_width": 4, "loop_count": 1}}


def fit_pin():
    import datagoblin.experiments.ldt_array_sorter as ldt

    return digest({p.name: file_hash(p) for p in [Path(__file__), HERE / "ALLOCATION_MODELS.md", Path(ldt.__file__)]})


def route(row, order, budget=PRIMARY, inference=0.0, use_features=False):
    elapsed = row["gate_s"] + inference + (row["features_s"] if use_features else 0)
    kernels = steps = work = 0
    tried = []
    ok = False
    end = "gate" if not row["gate"] else "exhausted"

    def charge(item):
        nonlocal elapsed, kernels, work
        work += item["work"]
        elapsed += item["solver_seconds"]
        if elapsed > budget:
            return False
        record = item["kernel"]
        if record is not None:
            if kernels >= 3:
                return False
            kernels += 1
            elapsed += record["seconds"]
        return elapsed <= budget

    if row["gate"]:
        for kind in order[:4]:
            if kernels >= 3 or elapsed >= budget:
                end = "budget"
                break
            tried.append(kind)
            if kind == "feedback":
                paid = True
                for source in row["source"]:
                    if not charge(source):
                        paid = False
                        break
                if not paid:
                    end = "budget"
                    break
            item = row["actions"][kind]
            if not charge(item):
                end = "budget"
                break
            if item["ok"]:
                ok = True
                end = "accepted"
                steps = item["steps"]
                break
    return dict(
        ok=ok,
        seconds=min(elapsed, budget),
        uncensored=elapsed,
        kernels=kernels,
        steps=steps,
        work=work,
        actions=tried,
        end=end,
    )


def objective(result, truth=True, budget=PRIMARY):
    return result["seconds"] + (2 * budget if truth and not result["ok"] else 0)


def acquisition(rows):
    kernels = {}
    solvers = sum(r["gate_s"] + r["features_s"] for r in rows)
    seen = set()
    for r in rows:
        trials = list(r["actions"].values())
        if r["lineage"] not in seen:
            trials += r["source"]
            seen.add(r["lineage"])
        for item in trials:
            solvers += item["solver_seconds"]
            k = item["kernel"]
            if k:
                kernels[k["source"]] = k["seconds"]
        if r["stale"]:
            for k in r["stale"].values():
                kernels[k["source"]] = k["seconds"]
    return dict(
        solver_seconds=solvers,
        kernel_seconds=sum(kernels.values()),
        unique_kernel_calls=len(kernels),
        total=solvers + sum(kernels.values()),
    )


def best_order(rows):
    return min(ORDERS, key=lambda order: sum(objective(route(r, order), r["gate"]) for r in rows))


def train_symbolic(rows):
    begin = time.perf_counter()
    best = best_order(rows)
    best_s = time.perf_counter() - begin
    begin = time.perf_counter()
    choices = []
    for i in (0, 6, 27, 28, 29):
        for threshold in (0.25, 0.5, 0.75):
            lo = [r for r in rows if r["features"][i] <= threshold]
            hi = [r for r in rows if r["features"][i] > threshold]
            if not lo or not hi:
                continue
            orders = [best_order(lo), best_order(hi)]
            score = sum(objective(route(r, orders[int(r["features"][i] > threshold)])) for r in rows)
            choices.append((score, i, threshold, orders))
    _, i, t, orders = min(choices)
    return dict(fixed=best, fixed_s=best_s, index=i, threshold=t, orders=orders, symbolic_s=time.perf_counter() - begin)


def mlp(width):
    return nn.Sequential(nn.Linear(width, 24), nn.GELU(), nn.Linear(24, len(ACTIONS)))


def ldt(width, seed):
    return make_model(WEIGHTS["architecture"], width + 2 * len(ACTIONS), seed, preference_count=2, rationale_count=2)


def pair_vector(x, i, j):
    return array("d", [*x, *(float(k == i) for k in range(4)), *(float(k == j) for k in range(4))])


def trained(rows, out):
    positive = [r for r in rows if r["gate"]]
    x = np.asarray([r["features"] for r in positive], dtype=float)
    mean = x.mean(0)
    scale = np.maximum(0.1, x.std(0))
    x = (x - mean) / scale
    targets = [[route(r, [a]) for a in ACTIONS] for r in positive]
    cost = np.array([[objective(z) for z in zs] for zs in targets])
    examples = []
    for k, xi in enumerate(x):
        for i, j in itertools.permutations(range(4), 2):
            pref = int((cost[k, j], j) < (cost[k, i], i))
            why = int(targets[k][i]["ok"] != targets[k][j]["ok"])
            examples.append((pair_vector(xi, i, j), pref, why))
    tensor = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(np.log1p(cost), dtype=torch.float32)
    fits = []
    for seed in SEEDS:
        for kind in ("mlp", "ldt"):
            start = time.perf_counter()
            if kind == "mlp":
                torch.manual_seed(seed)
                model = mlp(x.shape[1])
                opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01)
                for _ in range(240):
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.smooth_l1_loss(model(tensor), y)
                    loss.backward()
                    opt.step()
                ck = out / f"mlp-{seed}.pt"
                torch.save(model.state_dict(), ck)
                params = sum(p.numel() for p in model.parameters())
            else:
                model = ldt(x.shape[1], seed)
                for _ in range(180):
                    model.train_batch(
                        examples,
                        learning_rate=0.05,
                        l2=0.0001,
                        gradient_clip=2,
                        preference_class_weights=[1, 1],
                        rationale_class_weights=[1, 1],
                        rationale_loss_weight=0.15,
                    )
                ck = out / f"ldt-{seed}.json"
                save(ck, model.state_dict())
                params = model.parameter_count
            fits.append(
                dict(
                    kind=kind,
                    seed=seed,
                    params=params,
                    training_seconds=time.perf_counter() - start,
                    checkpoint=ck.name,
                    sha256=file_hash(ck),
                )
            )
    return mean.tolist(), scale.tolist(), fits


def freeze(out):
    train = read(out / "train.json")
    if train["split"] != "train" or train["pin"] != pin():
        raise ValueError("train source contract mismatch")
    if (out / "FROZEN.json").exists():
        raise ValueError("freeze already exists; use an independent output directory to refit")
    controls = train_symbolic([r for r in train["rows"] if r["gate"]])
    mean, scale, fits = trained(train["rows"], out)
    save(
        out / "FROZEN.json",
        dict(
            primitive_pin=pin(),
            fit_pin=fit_pin(),
            train_sha256=file_hash(out / "train.json"),
            controls=controls,
            mean=mean,
            scale=scale,
            fits=fits,
            acquisition=acquisition(train["rows"]),
        ),
    )
    print("Frozen", file_hash(out / "FROZEN.json"), flush=True)


def load_models(out, fr):
    result = {}
    for s in fr["fits"]:
        f = out / s["checkpoint"]
        if file_hash(f) != s["sha256"]:
            raise ValueError("checkpoint mutation")
        if s["kind"] == "mlp":
            m = mlp(len(fr["mean"]))
            m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
            m.eval()
        else:
            m = ldt(len(fr["mean"]), s["seed"])
            state = read(f)
            for k in (
                "input_weights",
                "hidden_bias",
                "preference_weights",
                "preference_bias",
                "rationale_weights",
                "rationale_bias",
            ):
                setattr(m, k, np.array(state[k]))
        result[f"{s['kind']}_{s['seed']}"] = m
    return result


def choose_model(kind, model, x):
    if kind == "mlp":
        with torch.no_grad():
            order = model(torch.tensor(x, dtype=torch.float32)).argsort(stable=True).tolist()
    else:
        win = np.zeros(4)
        for i, j in itertools.permutations(range(4), 2):
            p, _ = model.predict_proba(pair_vector(x, i, j))
            win[i] += p[0] / 2
            win[j] += p[1] / 2
        order = np.argsort(-win, kind="stable").tolist()
    return [ACTIONS[i] for i in order]


def summarize(rows, vals, training):
    positives = sum(r["gate"] for r in rows)
    sums = {key: sum(v[key] for v in vals) for key in ("seconds", "uncensored", "kernels", "steps", "work")}
    lineages = {}
    for r, v in zip(rows, vals, strict=True):
        lineages.setdefault(r["lineage"], {})[r["intervention"]] = v["ok"] == r["gate"]
    complete = [g for g in lineages.values() if len(g) == 3]
    return dict(
        positives=positives,
        complete_interventions=sum(all(g.values()) for g in complete),
        eligible_intervention_sets=len(complete),
        solved=sum(v["ok"] and r["gate"] for v, r in zip(vals, rows, strict=True)),
        false_accept=sum(v["ok"] and not r["gate"] for v, r in zip(vals, rows, strict=True)),
        timeouts=sum(v["end"] == "budget" for v in vals),
        training_seconds=training,
        all_in_seconds=training + sums["seconds"],
        **sums,
    )


def score(out):
    fr, held = read(out / "FROZEN.json"), read(out / "held.json")
    if (
        fr["fit_pin"] != fit_pin()
        or fr["primitive_pin"] != pin()
        or held["freeze_sha256"] != file_hash(out / "FROZEN.json")
        or fr["train_sha256"] != file_hash(out / "train.json")
    ):
        raise ValueError("frozen scoring contract changed")
    rows = held["rows"]
    models = load_models(out, fr)
    controls = fr["controls"]
    names = {a: ([a], False, 0.0) for a in ACTIONS}
    for order in ORDERS:
        names["fixed_" + "_".join(order)] = (order, False, 0.0)
    cost = fr["acquisition"]["total"]
    names["train_static"] = (controls["fixed"], False, cost + controls["fixed_s"])
    names["train_symbolic"] = (None, True, cost + controls["symbolic_s"])
    for fit in fr["fits"]:
        names[f"{fit['kind']}_{fit['seed']}"] = (None, True, cost + fit["training_seconds"])
    inference = {}
    choice = {}
    for name, (fixed, _needs_features, _) in names.items():
        choice[name] = []
        inference[name] = []
        for r in rows:
            start = time.perf_counter()
            if fixed is not None:
                order = fixed
            elif name == "train_symbolic":
                order = controls["orders"][int(r["features"][controls["index"]] > controls["threshold"])]
            elif r["gate"]:
                x = (np.asarray(r["features"]) - fr["mean"]) / fr["scale"]
                order = choose_model(name.split("_")[0], models[name], x)
            else:
                order = ()
            inference[name].append(time.perf_counter() - start)
            choice[name].append(order)
    tables = {}
    details = {}
    for multiple in (2, 4, 8):
        budget = MEDIAN * multiple
        table = {}
        for name, (_, info, t) in names.items():
            runs = [
                route(r, o, budget, inf, info and r["gate"])
                for r, o, inf in zip(rows, choice[name], inference[name], strict=True)
            ]
            one = summarize(rows, runs, t)
            one["inference_seconds"] = sum(inference[name])
            one["families"] = {
                f: summarize(
                    [r for r in rows if r["family"] == f],
                    [v for r, v in zip(rows, runs, strict=True) if r["family"] == f],
                    0,
                )
                for f in sorted({r["family"] for r in rows})
            }
            one["interventions"] = {
                f: summarize(
                    [r for r in rows if r["intervention"] == f],
                    [v for r, v in zip(rows, runs, strict=True) if r["intervention"] == f],
                    0,
                )
                for f in ("unchanged", "edited", "invalid")
            }
            table[name] = one
            if multiple == 4:
                details[name] = runs
        oracle = [
            min((route(r, o, budget) for o in ORDERS), key=lambda v: objective(v, r["gate"], budget)) for r in rows
        ]
        table["oracle"] = summarize(rows, oracle, 0.0)
        tables[str(multiple)] = table
    pri = tables["4"]
    symbolic = [k for k in names if k not in models]
    best = min(symbolic, key=lambda n: (-pri[n]["solved"], pri[n]["all_in_seconds"], n))
    validations = [r["stale"] for r in rows if r["intervention"] != "unchanged"]
    transport_pass = sum(bool(v and v["context"]["accepted"] and v["invalid"]["rejected"]) for v in validations)

    def grouped(item):
        return sum(g["solved"] for f, g in item["families"].items() if f.startswith("mixed"))

    gate = {}
    for kind in ("mlp", "ldt"):
        checks = []
        for seed in SEEDS:
            v = pri[f"{kind}_{seed}"]
            ref = pri[best]
            checks.append(
                v["solved"] >= ref["solved"]
                and v["all_in_seconds"] <= 0.95 * ref["all_in_seconds"]
                and v["false_accept"] == 0
                and grouped(v) >= grouped(ref)
                and transport_pass == len(validations)
            )
        gate[kind] = all(checks)
    save(
        out / "POLICIES.json",
        dict(
            tables=tables,
            details=details,
            choices=choice,
            inference_seconds_by_row=inference,
            baseline=best,
            passes=gate,
            invalid_transports_checked=transport_pass,
            invalid_transports_expected=len(validations),
            train_acquisition=fr["acquisition"],
            fit_sha256=file_hash(out / "FROZEN.json"),
            held_sha256=file_hash(out / "held.json"),
            train_sha256=fr["train_sha256"],
        ),
    )
    print(best, gate, transport_pass, "/", len(validations), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "score"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    (freeze if args.mode == "freeze" else score)(args.out)


if __name__ == "__main__":
    main()
