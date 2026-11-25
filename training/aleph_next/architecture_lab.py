"""Finite verified repair and witness tasks for source access and magnitude-sensitive clocks."""

import argparse
import hashlib
import json
import math
import secrets
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
PLACEMENT = HERE.parent / "aleph_rlt_placement"
sys.path.insert(0, str(PLACEMENT))
from gym import Block, atomic_json, file_sha  # noqa: E402

ARMS = {
    "direct2": 2,
    "damp2": 2,
    "source2": 2,
    "local2": 2,
    "gate_source2": 2,
    "gate_local2": 2,
    "direct3": 3,
    "damp3": 3,
    "rlt3": 3,
    "magnitude3": 3,
    "visit3": 3,
}
LEARNED = ("gate_source2", "gate_local2", "magnitude3", "visit3")
TASKS = ("corrupted_proof", "wrong_rule", "inconsistent_steps", "misleading_solution")
HELD = ((0, 2), (2, 0), (1, 3), (3, 1))
TRAIN = tuple((a, b) for a in range(4) for b in range(4) if (a, b) not in HELD)
STRENGTH = (0.0, 0.5, 1.25)
CONDITIONS = (("clean", 0.0), ("target", 0.5), ("random", 0.5), ("target", 1.25), ("random", 1.25))
METRICS = (
    "bce",
    "verified",
    "cheap",
    "pass_verified",
    "activation_rms",
    "proposed_relative",
    "realized_relative",
    "update_cosine",
    "jvp_gain",
    "curvature",
    "clock",
    "reread",
    "injected_relative",
)


def op(a, b, tag):
    return (a ^ b, a & b, a | b, (15 ^ a) | b)[tag]


def sample(rng, task, pair):
    first, second = pair
    while True:
        a, b, c = map(int, rng.integers(16, size=3))
        h, wrong = op(a, b, first), op(a, b, (first + int(rng.integers(1, 4))) % 4)
        if task != 1 or h != wrong:
            break
    g = op(h, c, second)
    fault = int(rng.integers(1, 16))
    written_h = wrong if task in (1, 3) else h ^ fault
    working = op(written_h, c, second)
    if task == 2:
        written_g = working ^ fault
    elif task == 3:
        written_g = g ^ fault
    else:
        written_g = working
    truth = g if task < 2 else (working if task == 2 else g) ^ written_g
    return a, b, c, written_h, written_g, first, second, task, truth


def tensorize(rows):
    arr = np.asarray(rows)
    n = len(arr)
    features = np.zeros((n, 4, 17), dtype=np.float32)
    bit_ids = np.arange(4)
    for col in range(5):
        features[:, :, col] = 2 * ((arr[:, col, None] >> bit_ids) & 1) - 1
    for col, start in ((5, 5), (6, 9), (7, 13)):
        features[:, :, start : start + 4] = np.eye(4, dtype=np.float32)[arr[:, col], None, :]
    labels = ((arr[:, 8, None] >> bit_ids) & 1).astype(np.float32)
    return torch.from_numpy(features), torch.from_numpy(labels), torch.tensor(arr[:, 7])


def training_batch(seed, step, batch, width):
    rng = np.random.default_rng([seed, step, 812931])
    rows = [sample(rng, int(rng.integers(4)), TRAIN[rng.integers(len(TRAIN))]) for _ in range(batch)]
    x, y, task = tensorize(rows)
    kinds = torch.tensor(rng.integers(2, size=batch)).bool()
    levels = torch.tensor(rng.choice(STRENGTH, size=batch), dtype=torch.float32)
    random = torch.tensor(rng.standard_normal((batch, 4, width)), dtype=torch.float32)
    return x, y, task, kinds, levels, random


def rms(x):
    return x.square().mean(dim=(-1, -2), keepdim=True).clamp_min(1e-20).sqrt()


def relative(a, b):
    return rms(a) / rms(b)


def token_rms(x):
    return x.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()


class MiniAleph(nn.Module):
    def __init__(self, width, arm):
        super().__init__()
        self.embed = nn.Linear(17, width)
        self.prelude = Block(width)
        self.u = Block(width)
        self.v = Block(width)
        self.coda = Block(width)
        self.norm = nn.RMSNorm(width)
        self.out = nn.Linear(width, 1)
        self.arm = arm
        if arm in LEARNED:
            self.gate = nn.Parameter(torch.zeros(2))

    def source(self, features):
        return self.prelude(self.embed(features))

    def readout(self, state):
        return self.out(self.norm(self.coda(state))).squeeze(-1)

    def core(self, x, source, arm, visit):
        middle = self.u(x)
        beta = torch.zeros_like(x[:, :1, :1])
        if visit == 1 and arm in ("source2", "local2", "gate_source2", "gate_local2"):
            anchor = source if "source" in arm else x
            if arm.startswith("gate"):
                distance = token_rms(anchor - middle) / token_rms(middle)
                beta = (math.log(1 / 3) + self.gate[0] + self.gate[1] * distance.log1p()).sigmoid()
            else:
                beta = beta + 0.25
            middle = middle + beta * (anchor - middle)
        return self.v(middle), beta

    def alpha(self, arm, visit, raw, x):
        ones = torch.ones_like(x[:, :1, :1])
        if visit == ARMS[arm] - 1:
            return ones
        if arm.startswith("damp"):
            return ones * (0.5 if ARMS[arm] == 2 else 5 / 12)
        if arm == "rlt3":
            return ones / (3 - visit)
        if arm in ("magnitude3", "visit3"):
            statistic = (token_rms(raw - x) / token_rms(x)).log1p() if arm == "magnitude3" else ones * (visit - 0.5)
            return (math.log(5 / 7) + self.gate[0] + self.gate[1] * statistic).sigmoid()
        return ones

    def forward(self, features, kinds, levels, noise, arm=None, trace=False, differential=False):
        arm = arm or self.arm
        source = self.source(features)
        with torch.no_grad():
            foil = features.clone()
            foil[:, :, :2] = features[:, :, 3:5]
            foil[:, :, 2] *= -1
            targeted = self.source(foil) - source.detach()
            direction = torch.where(kinds[:, None, None], targeted, noise)
            unit = direction / token_rms(direction)
        x, previous = source, None
        stats = {k: [] for k in METRICS[3:]}
        readouts = []
        for visit in range(ARMS[arm]):
            raw, beta = self.core(x, source, arm, visit)
            alpha = self.alpha(arm, visit, raw, x)
            # All terminal and direct transports use the native endpoint exactly.
            end = (
                raw
                if visit == ARMS[arm] - 1
                or arm in ("direct2", "direct3", "source2", "local2", "gate_source2", "gate_local2")
                else x + alpha * (raw - x)
            )
            delta = end - x
            if trace:

                def keep(name, value):
                    stats[name].append(value.flatten(1).mean(1))

                keep("activation_rms", rms(end))
                keep("proposed_relative", relative(raw - x, x))
                keep("realized_relative", relative(delta, x))
                if previous is not None:
                    keep("update_cosine", F.cosine_similarity(previous.flatten(1), delta.flatten(1))[:, None])
                keep("clock", alpha)
                if visit == 1:
                    keep("reread", beta)
                if differential:
                    # A unit RMS direction, common across schedules and paired perturbation kinds.
                    v = noise.flip(-1) / rms(noise)
                    eps = 0.035 * rms(x)
                    plus = self.core(x + eps * v, source, arm, visit)[0]
                    minus = self.core(x - eps * v, source, arm, visit)[0]
                    keep("jvp_gain", rms((plus - minus) / (2 * eps)))
                    keep("curvature", rms((plus - 2 * raw + minus) / eps.square()))
            previous = delta
            if trace or visit == ARMS[arm] - 1:
                readouts.append(self.readout(end))
            if visit == 0:
                injection = levels[:, None, None] * token_rms(end.detach()) * unit
                if trace:
                    keep("injected_relative", (injection.square().mean((1, 2)).sqrt() / rms(end).flatten())[:, None])
                x = end + injection
            else:
                x = end
        return readouts, stats


def verdict(logits, labels, task):
    repair = ((logits >= 0) == labels.bool()).all(dim=1)
    witness = labels.gather(1, logits.argmax(1, keepdim=True)).squeeze(1).bool()
    return torch.where(task < 2, repair, witness).float()


def cheap_verdict(x, labels, task):
    copy = ((x[:, :, 4] > 0) == labels.bool()).all(dim=1)
    return torch.where(task < 2, copy, labels[:, 0].bool()).float()


def stream_digest(x, y, task, kinds, levels, random):
    d = hashlib.sha256()
    for v in (x, y, task, kinds, levels, random):
        d.update(v.numpy().tobytes())
    return d.digest()


def contract(cfg):
    return {
        "recipe": cfg,
        "sources": {
            f.name: hashlib.sha256(f.read_text(encoding="utf-8").encode()).hexdigest()
            for f in (Path(__file__), HERE / "ARCH_PROTOCOL.md", PLACEMENT / "gym.py")
        },
        "torch": str(torch.__version__),
    }


def fit_one(arm, seed, cfg, directory):
    width, batch, steps = (cfg[k] for k in ("width", "batch", "steps"))
    torch.manual_seed(seed)
    model = MiniAleph(width, arm)
    init = hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for n, p in model.named_parameters() if n != "gate")
    ).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    rss = psutil.Process()
    peak = rss.memory_info().rss
    losses, grads, stream = [], [], hashlib.sha256()
    start = time.monotonic()
    with torch.no_grad():
        params = sum(p.numel() for p in model.parameters())
    for step in range(steps):
        data = training_batch(seed, step, batch, width)
        stream.update(stream_digest(*data))
        x, y, _, kinds, levels, noise = data
        opt.zero_grad(set_to_none=True)
        if step == 0:
            with FlopCounterMode(display=False) as counter:
                logits = model(x, kinds, levels, noise)[0][-1]
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
            flops = int(counter.get_total_flops())
            flop_ops = {str(op): int(v) for op, v in counter.get_flop_counts()["Global"].items()}
        else:
            logits = model(x, kinds, levels, noise)[0][-1]
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % 16 == 0 or step == steps - 1:
            peak = max(peak, rss.memory_info().rss)
            losses.append([step + 1, float(loss.detach())])
            grads.append(float(grad))
    seconds = time.monotonic() - start
    path = directory / f"{arm}-{seed}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)
    return {
        "seed": seed,
        "arm": arm,
        "positions": steps * batch * 4,
        "parameters": params,
        "trainable_gate_parameters": 2 if arm in LEARNED else 0,
        "backbone_init": init,
        "stream_sha256": stream.hexdigest(),
        "checkpoint_sha256": file_sha(path),
        "wall_seconds": seconds,
        "positions_per_second": steps * batch * 4 / seconds,
        "sampled_peak_rss_mib": peak / 2**20,
        "core_visits": ARMS[arm],
        "native_blocks_per_example": 2 + 2 * ARMS[arm],
        "extra_no_grad_foil_prelude_blocks": 1,
        "first_update_counted_flops": flops,
        "counted_step_ops": flop_ops,
        "counted_flops_all_updates": flops * steps,
        "training_bce": losses,
        "sampled_gradient_norm": grads,
        "learned_gate": model.gate.detach().tolist() if arm in LEARNED else None,
    }


def train(cfg, out):
    path = out / "training.json"
    spec = contract(cfg)
    result = json.loads(path.read_text()) if path.exists() else {"contract": spec, "fits": {}}
    if result["contract"] != spec:
        raise ValueError("fit code, protocol or recipe changed")
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            key = f"{arm}-{seed}"
            if key in result["fits"]:
                if file_sha(out / f"{key}.pt") != result["fits"][key]["checkpoint_sha256"]:
                    raise ValueError("checkpoint digest changed")
                continue
            row = fit_one(arm, seed, cfg, out)
            result["fits"][key] = row
            atomic_json(path, result)
            print(key, row["parameters"], round(row["wall_seconds"], 2), flush=True)


def evaluation_rows(salt, width, n):
    rng = np.random.default_rng(salt)
    rows = []
    groups = []
    for pair in HELD:
        for task in range(4):
            rows.extend(sample(rng, task, pair) for _ in range(n))
            groups.extend([f"{pair[0]}-{pair[1]}/{TASKS[task]}"] * n)
    x, y, task = tensorize(rows)
    noise = torch.tensor(rng.standard_normal((len(x), 4, width)), dtype=torch.float32)
    return x, y, task, noise, groups


@torch.inference_mode()
def evaluate(model, panel, arm, probes):
    x, y, t, noise, groups = panel
    out = {}
    for kind, strength in CONDITIONS:
        metrics = {}
        for start in range(0, len(x), 32):
            sl = slice(start, start + 32)
            k = torch.full((len(x[sl]),), kind == "target")
            levels = torch.full((len(x[sl]),), strength)
            log, stats = model(x[sl], k, levels, noise[sl], arm, trace=True, differential=probes)
            hits = [verdict(z, y[sl], t[sl]) for z in log]
            pieces = {
                "bce": F.binary_cross_entropy_with_logits(log[-1], y[sl], reduction="none").mean(1)[:, None],
                "verified": hits[-1][:, None],
                "cheap": cheap_verdict(x[sl], y[sl], t[sl])[:, None],
                "pass_verified": torch.stack(hits, 1),
            }
            if probes:
                pieces.update({n: torch.stack(v, 1) for n, v in stats.items() if v})
            for name, value in pieces.items():
                metrics.setdefault(name, []).append(value)
        gathered = {n: torch.cat(v).numpy() for n, v in metrics.items()}
        results = {}
        a = np.array(groups)
        for group in sorted(set(groups)):
            idx = a == group
            results[group] = {n: values[idx].mean(axis=0, dtype=np.float64).tolist() for n, values in gathered.items()}
        out[f"{kind}-{strength}"] = results
    return out


def score(out):
    path = out / "training.json"
    record = json.loads(path.read_text())
    cfg = record["contract"]["recipe"]
    if record["contract"] != contract(cfg):
        raise ValueError("scorer differs from frozen code")
    expected = {f"{a}-{s}" for s in cfg["seeds"] for a in cfg["arms"]}
    if set(record["fits"]) != expected:
        raise ValueError("complete the registered fits before generating the held panel")
    for key, row in record["fits"].items():
        if file_sha(out / f"{key}.pt") != row["checkpoint_sha256"]:
            raise ValueError("invalid checkpoint")
    target = out / "private.seed"
    if not target.exists():
        with target.open("x", encoding="ascii") as stream:
            stream.write(str(secrets.randbits(128)) + "\n")
    panel = evaluation_rows(int(target.read_text()), cfg["width"], 16)
    output = {
        "training_sha256": file_sha(path),
        "seed_sha256": file_sha(target),
        "examples": len(panel[0]),
        "native": {},
        "crossovers": {},
    }
    for key, row in record["fits"].items():
        arm = row["arm"]
        m = MiniAleph(cfg["width"], arm).eval()
        m.load_state_dict(torch.load(out / f"{key}.pt", map_location="cpu", weights_only=True))
        output["native"][key] = evaluate(m, panel, arm, True)
        same = ("rlt3", "damp3", "direct3") if ARMS[arm] == 3 else ("direct2", "source2", "local2")
        output["crossovers"][key] = {other: evaluate(m, panel, other, False) for other in same if other != arm}
        print("scored", key, flush=True)
    dest = out / "scores.json"
    if dest.exists() and json.loads(dest.read_text()) != output:
        raise ValueError("score ledger changed")
    atomic_json(dest, output)
    export_receipts(record, output, out)


def avg_conditions(data, conditions, selected=None, metric="verified"):
    vals = []
    for con in conditions:
        for family, row in data[con].items():
            if selected is None or selected(family):
                vals.append(row[metric][0])
    return mean(vals)


def export_receipts(record, output, out):
    cfg = record["contract"]["recipe"]
    summary = {}
    high = ("target-1.25", "random-1.25")
    clean = ("clean-0.0",)
    conditions = tuple(output["native"][next(iter(record["fits"]))])
    for key in record["fits"]:
        data = output["native"][key]
        row = {
            "high": avg_conditions(data, high),
            "clean": avg_conditions(data, clean),
            "clean_bce": avg_conditions(data, clean, metric="bce"),
            "high_bce": avg_conditions(data, high, metric="bce"),
            "target": avg_conditions(data, [high[0]]),
            "random": avg_conditions(data, [high[1]]),
        }
        for group, subset in (("repair", TASKS[:2]), ("witness", TASKS[2:])):

            def keep(f, subset=subset):
                return f.rsplit("/", 1)[1] in subset

            row[group] = avg_conditions(data, conditions, keep)
            row[group + "_cheap"] = avg_conditions(data, conditions, keep, "cheap")
        row["compositions"] = {
            f"{a}-{b}": avg_conditions(data, high, lambda f, tag=f"{a}-{b}/": f.startswith(tag)) for a, b in HELD
        }
        summary[key] = row
    gates = {}
    seeds = cfg["seeds"]

    def lift(a, b, metric):
        return [summary[f"{a}-{s}"][metric] - summary[f"{b}-{s}"][metric] for s in seeds]

    for cand, rivals, strong in (
        ("source2", ("direct2", "local2"), "local2"),
        ("gate_source2", ("source2", "gate_local2"), "gate_local2"),
        ("rlt3", ("damp3",), "damp3"),
        ("magnitude3", ("damp3", "visit3"), "visit3"),
    ):
        direct = f"direct{ARMS[cand]}"
        if not {cand, direct, *rivals}.issubset(cfg["arms"]):
            continue
        competent = {
            t: mean(summary[f"{direct}-{s}"][t] - summary[f"{direct}-{s}"][t + "_cheap"] for s in seeds) >= 0.05
            for t in ("repair", "witness")
        }
        hi = {r: lift(cand, r, "high") for r in rivals}
        better_families = sum(
            mean(summary[f"{cand}-{s}"]["compositions"][f] - summary[f"{strong}-{s}"]["compositions"][f] for s in seeds)
            > 0
            for f in summary[f"{cand}-{seeds[0]}"]["compositions"]
        )
        if ARMS[cand] == 2:
            depth_gate = mean(lift(cand, direct, "clean")) >= -0.02
        else:
            depth_gate = mean(lift(cand, "damp3", "high")) - mean(lift(cand, "damp3", "clean")) >= 0.01
        enough_margin = all(
            min(lifts) > 0 and mean(lifts) >= 0.02
            for name, lifts in hi.items()
            if not (cand == "magnitude3" and name == "visit3")
        )
        checks = {
            "direct_competence": all(competent.values()),
            "rival_margin": enough_margin,
            "wins_all_seeds": all(min(x) > 0 for x in hi.values()),
            "depth_specific": depth_gate,
            "three_families": better_families >= 3,
            "both_directions": all(mean(lift(cand, strong, k)) > 0 for k in ("target", "random")),
        }
        gates[cand] = {
            "passed": all(checks.values()),
            "checks": checks,
            "direct_competence_by_task": competent,
            "high_seed_lifts": hi,
            "positive_compositions": better_families,
        }
    atomic_json(out / "summary.json", {"cells": summary, "gates": gates})
    with (out / "integration.jsonl").open("w", encoding="utf-8") as fp:
        for key, fit in record["fits"].items():
            row = {
                "id": f"aleph-arch-v1/w{cfg['width']}-{fit['positions']}/{key}",
                "gym": "aleph-01-02-boolean-repair",
                "model": fit,
                "recipe": cfg,
                "source_sha256": record["contract"]["sources"],
                "verifier": "architecture_lab.op and binary truth-table verdict; held answers excluded",
                "benchmark": summary[key],
                "evaluation_protocol": "ARCH_PROTOCOL.md",
                "mechanistic_high": {
                    k: np.mean([r[k] for c in high for r in output["native"][key][c].values()], axis=0).tolist()
                    for k in METRICS[3:]
                    if k in next(iter(output["native"][key][high[0]].values()))
                },
                "communication_transfer": "No message channel. Unseen rule compositions at matched corruption.",
                "known_limitations": [
                    "Four positions and two shared middle blocks",
                    "Finite Boolean verifier",
                    "Four held compositions",
                ],
            }
            fp.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("train", "score"))
    cli.add_argument("--width", type=int, default=32)
    cli.add_argument("--batch", type=int, default=32)
    cli.add_argument("--steps", type=int, default=192)
    cli.add_argument("--seeds", nargs="+", type=int, default=[137, 163, 191])
    cli.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.width % 8 or min(args.width, args.batch, args.steps) <= 0:
        cli.error("positive dimensions and a width divisible by eight are required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        train({k: getattr(args, k) for k in ("width", "batch", "steps", "seeds", "arms")}, args.out)
    else:
        score(args.out)
