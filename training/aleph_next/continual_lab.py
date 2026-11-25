"""Three-skill continual learning with exact 3/4-bit parity oracles and native LDT."""

import argparse
import copy
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from receipt_coordination import build_teacher
from receipt_gym import SEEDS, TRAIN_PRIORS, Game, controls, digest, parity, score
from receipt_models import ReceiptModel
from receipt_relations import make_relations
from torch import nn
from torch.nn import functional as F

METHODS = ("sequential", "interleaved", "replay", "distill", "project", "route", "heads")
P4 = (0.0, 0.01, 0.02, 0.03, 0.05, 0.25, 1.0)
BRACKETS = {3: (0.07, 0.072, 0.079, 0.08), 4: (0.021, 0.023, 0.025, 0.027)}


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def four_oracle():
    directions = [sum(1 << x for x in range(16) if (s & x).bit_count() % 2) for s in range(1, 16)]
    first = directions[0]
    groups = np.zeros(32768, dtype=np.int64)
    moment = np.zeros_like(groups)
    pop = np.array([i.bit_count() for i in range(65536)])
    mask = np.arange(0, 65536, 2)
    for split in directions:
        for half in (split, 65535 ^ split):
            cell = half & mask
            for part in (cell, half ^ cell):
                n = pop[part]
                groups += n > 0
                moment += (n - 2 * pop[first & part]) ** 2 * (840 // np.maximum(n, 1))
    # Four times gain over an uninformed forecast, for uniform and first-bit questions.
    lo = (groups - 15) / 225
    hi = moment / (16 * 15 * 840)
    at_train = 0.99 * lo + 0.01 * hi
    candidates = np.flatnonzero(at_train >= at_train.max() - 1e-12)
    lower = min((2 * int(i) for i in candidates), key=lambda m: ((m ^ first).bit_count(), m))

    def exact(i):
        return Fraction(int(groups[i] - 15), 225), Fraction(int(moment[i]), 16 * 15 * 840)

    a, b = exact(lower // 2)
    c, d = exact(first // 2)
    cross = (a - c) / (d - c - b + a)
    assert cross == Fraction(16, 667)
    # All three exact cuts share integer denominators across codes.
    assert groups[lower // 2] == max(groups) and moment[first // 2] == max(moment)
    common = (groups - 15) * (cross.denominator - cross.numerator) * 16 * 15 * 840 + moment * cross.numerator * 225
    assert common[lower // 2] == common[first // 2] == max(common)
    return (1 - lo) / 4, (1 - hi) / 4, lower, first


class RoutedScore(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.heads = nn.ModuleDict({k: copy.deepcopy(head) for k in ("p3", "ldt", "p4")})
        self.task = "p3"

    def forward(self, x):
        return self.heads[self.task](x)


class Student(ReceiptModel):
    def __init__(self, method, parent):
        super().__init__("C")
        self.load_state_dict(parent)
        self.method = method
        self.task = "p3"
        if method == "heads":
            self.score = RoutedScore(self.score)
        if method == "route":
            self.requires_grad_(False)
            self.adapters = nn.ModuleDict({k: nn.Linear(12, 12) for k in ("ldt", "p4")})
            for adapter in self.adapters.values():
                nn.init.zeros_(adapter.weight)
                nn.init.zeros_(adapter.bias)

    def select(self, task):
        self.task = task
        if self.method == "heads":
            self.score.task = task

    def _features(self, features):
        x = super()._features(features)
        return x + self.adapters[self.task](x) if self.method == "route" and self.task != "p3" else x


def game4(p):
    raw = parity(0)[0]
    xs = np.zeros((16, 12))
    xs[:, :3] = np.tile(raw[:, :3], (2, 1))
    xs[:, 3] = p
    xs[:, 7] = 2 * ((np.arange(16) >> 3) & 1) - 1
    xs[:, 8] = 1
    t = np.repeat(np.arange(1, 16), 15)
    s = np.tile(np.arange(1, 16), 15)
    r = np.zeros((225, 12))
    for offset, direction in ((0, t), (3, s)):
        r[:, offset : offset + 3] = 2 * ((direction[:, None] >> np.arange(3)) & 1) - 1
    r[:, 6] = p
    r[:, 7] = 2 * ((t >> 3) & 1) - 1
    r[:, 8] = 2 * ((s >> 3) & 1) - 1
    r[:, 10] = 1
    parity_lut = np.array([x.bit_count() % 2 for x in range(16)])
    y = parity_lut[np.arange(16)[:, None] & t]
    side = parity_lut[np.arange(16)[:, None] & s]
    w = ((1 - p) / 15 + p * (t == 1)) / 15
    return xs, r, y, side, w


def bits(mask, n):
    return torch.tensor([(mask >> x) & 1 for x in range(n)], dtype=torch.float)


def receiver4(model, raw, code, indices=None):
    _, r, y, s, w = raw
    q = np.arange(len(w)) if indices is None else indices
    weights = torch.tensor(w if indices is None else np.full(len(q), 1 / len(q)), dtype=torch.float)
    side = torch.tensor(s[:, q])
    yy = torch.tensor(y[:, q])
    n = len(q)
    sf = torch.arange(2).repeat_interleave(n).repeat(2).float()
    msg = torch.arange(2).repeat_interleave(2 * n).float()
    guess = model.decode(torch.tensor(r[q], dtype=torch.float).repeat(4, 1), sf, msg).sigmoid().reshape(2, 2, n)
    chosen = guess[code.long()[:, None], side, torch.arange(n)]
    return ((chosen - yy).square() * weights).sum() / 16


def setup():
    o4 = four_oracle()
    th = build_teacher()
    endpoints = {3: (controls(parity(0))[0], controls(parity(1))[0]), 4: o4[:2]}
    old = [Game(parity(p)) for p in TRAIN_PRIORS]
    oldlogits = torch.tensor(np.stack([th.target_logits(p) for p in TRAIN_PRIORS]), dtype=torch.float)
    oldbits = [torch.tensor(th.hard_code(p), dtype=torch.float) for p in TRAIN_PRIORS]
    four = [game4(p) for p in P4]
    a, b = o4[2:]
    slope = (o4[1][a // 2] - o4[1][b // 2]) - (o4[0][a // 2] - o4[0][b // 2])
    t4 = []
    c4 = []
    for p in P4:
        risk = (1 - p) * o4[0] + p * o4[1]
        margin = risk[a // 2] - risk[b // 2]
        t4.append(torch.where(bits(a, 16) == bits(b, 16), 2 * bits(a, 16) - 1, (2 * bits(b, 16) - 1) * margin / slope))
        c4.append(bits(b if margin > 0 else a, 16))
    return dict(
        endpoints=endpoints,
        old=old,
        t3=oldlogits,
        c3=oldbits,
        four=four,
        t4=torch.stack(t4),
        c4=c4,
        ldt=[Game(g) for g in make_relations("train", 23017).values()],
    )


def rehearsal(model, ctx, j):
    model.select("p3")
    g = ctx["old"][j]
    pred = model.encode(g.p)
    goal = ctx["t3"][j]
    return (
        F.smooth_l1_loss((pred - goal) / goal.abs().clamp_min(0.05), torch.zeros_like(goal))
        + g.risks(ctx["c3"][j], g.guesses(model))[0]
    )


def ldt_loss(model, ctx, i):
    model.select("ldt")
    g = ctx["ldt"][i]
    real, ideal = g.risks(model.encode(g.p).sigmoid(), g.guesses(model))
    return (real + ideal) / 2


def distillation(model, teacher, g, task):
    model.select(task)
    teacher.select(task)
    with torch.no_grad():
        target = teacher.encode(g.p)
        post = g.guesses(teacher)
    return (model.encode(g.p) - target).square().mean() + (g.guesses(model) - post).square().mean()


@torch.no_grad()
def probe(model, ctx, d):
    model.select("p3" if d == 3 else "p4")
    priors = np.linspace(0.05, 0.10, 201) if d == 3 else np.linspace(0.012, 0.036, 201)
    base = (parity(0) if d == 3 else game4(0))[0]
    xs = np.repeat(base[None], len(priors), axis=0)
    xs[:, :, 3] = priors[:, None]
    n = len(base)
    code = (model.encode(torch.tensor(xs.reshape(-1, 12), dtype=torch.float)) >= 0).reshape(-1, n).int().numpy()
    masks = (code ^ code[:, :1]) @ (1 << np.arange(n))
    a, b = ctx["endpoints"][d]
    if d == 4:
        # Endpoints select two envelope segments; cross was proved by four_oracle.
        # Take the true all-code minimum in bounded batches.
        minrisk = np.array([np.min((1 - p) * a + p * b) for p in priors])
    else:
        minrisk = ((1 - priors[:, None]) * a + priors[:, None] * b).min(1)
    regret = (1 - priors) * a[masks // 2] + priors * b[masks // 2] - minrisk
    bracket = []
    for p in BRACKETS[d]:
        raw = (parity if d == 3 else game4)(p)
        z = (model.encode(torch.tensor(raw[0], dtype=torch.float)) >= 0).float()
        mask = int((z.int().numpy() ^ int(z[0])) @ (1 << np.arange(n)))
        risk = (1 - p) * a + p * b
        if d == 3:
            g = Game(raw)
            actual = float(g.risks(z, g.guesses(model))[0])
        else:
            actual = float(receiver4(model, raw, z))
        bracket.append(
            dict(
                p=p,
                mask=mask,
                optimal=bool(risk[mask // 2] <= risk.min() + 1e-10),
                brier=actual,
                bit_bayes=float(risk[mask // 2]),
                side=float(risk[0]),
            )
        )
    return dict(
        optimal_fraction=float(np.mean(regret < 1e-10)),
        mean_regret=float(np.mean(regret)),
        bracket=bracket,
        code_pass=all(r["optimal"] for r in bracket),
        pass_all=all(r["optimal"] and r["brier"] < r["side"] for r in bracket),
    )


def fit_stage(model, ctx, stage, method, seed):
    prior = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    rng = np.random.default_rng(seed + (stage == "p4") * 700)
    auxrng = np.random.default_rng(seed + 409)
    steps = 240 if stage == "ldt" else 180
    trace = []
    inputs = torch.tensor(np.stack([raw[0] for raw in ctx["four"]]), dtype=torch.float).reshape(-1, 12)
    for step in range(steps):
        j = int(rng.integers(len(ctx["ldt"] if stage == "ldt" else P4)))
        if stage == "ldt":
            loss = ldt_loss(model, ctx, j)
        else:
            model.select("p4")
            raw = ctx["four"][j]
            subset = rng.choice(225, 24, p=raw[-1], replace=True)
            pred = model.encode(inputs).reshape_as(ctx["t4"])
            delta = (pred - ctx["t4"]) / ctx["t4"].abs().clamp_min(0.025)
            loss = F.smooth_l1_loss(delta, torch.zeros_like(delta)) + receiver4(model, raw, ctx["c4"][j], subset)
        obj = loss
        opt.zero_grad(set_to_none=True)
        if method in ("interleaved", "replay", "distill", "project"):
            use_ldt = stage == "p4" and step % 2 == 1
            m = int(auxrng.integers(len(ctx["ldt"] if use_ldt else TRAIN_PRIORS)))
            if method == "replay":
                m = (step // 2) % 2 if use_ldt else (2, 3)[(step // 2) % 2]
            task = "ldt" if use_ldt else "p3"

            def previous(use_ldt=use_ldt, m=m):
                return ldt_loss(model, ctx, m) if use_ldt else rehearsal(model, ctx, m)

            if method == "interleaved":
                obj.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                obj = previous()
            elif method == "distill":
                obj = obj + 10 * distillation(model, prior, ctx["ldt" if use_ldt else "old"][m], task)
            elif method == "replay":
                obj = obj + previous()
            else:
                params = list(model.parameters())
                ref = torch.autograd.grad(previous(), params, allow_unused=True)
                obj.backward()
                pairs = [(p, g) for p, g in zip(params, ref, strict=True) if g is not None and p.grad is not None]
                dot = sum((p.grad * g).sum() for p, g in pairs)
                norm = sum(g.square().sum() for _, g in pairs)
                alpha = dot.clamp_max(0) / norm.clamp_min(1e-12)
                for p, g in pairs:
                    p.grad.sub_(alpha * g)
        if method != "project":
            obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step + 1 in (steps // 2, steps):
            trace.append(
                dict(step=step + 1, p3=probe(model, ctx, 3), p4=probe(model, ctx, 4) if stage == "p4" else None)
            )
    return trace


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("train", "score"))
    cli.add_argument("--base", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--seed-file", type=Path)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ctx = setup()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.base / "training.json").read_text())["runs"]
    manifest = out / "training.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes() + Path(__file__).with_name("CONTINUAL_PROTOCOL.md").read_bytes()
    ).hexdigest()
    if args.phase == "train":
        for seed in SEEDS:
            parent = args.base / f"C-{seed}-coordinated.pt"
            assert digest(parent) == base[f"C-{seed}"]["coordinated"]["sha256"]
            for method in METHODS:
                key = f"{method}-{seed}"
                torch.manual_seed(seed)
                model = Student(method, torch.load(parent, weights_only=True))
                for stage in ("ldt", "p4"):
                    name = f"{key}-{stage}"
                    ck = out / (name + ".pt")
                    if name in data:
                        if digest(ck) != data[name]["checkpoint_sha256"] or data[name]["contract"] != fingerprint:
                            raise ValueError("partial cell mismatch")
                        model.load_state_dict(torch.load(ck, weights_only=True))
                        continue
                    start = time.perf_counter()
                    initial = probe(model, ctx, 3)
                    tr = fit_stage(model, ctx, stage, method, seed)
                    torch.save(model.state_dict(), ck)
                    data[name] = dict(
                        arm=method,
                        stage=stage,
                        seed=seed,
                        params=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        updates=(240 if stage == "ldt" else 180) * (1 + int(method == "interleaved")),
                        checkpoint_sha256=digest(ck),
                        parent_sha256=digest(parent) if stage == "ldt" else data[key + "-ldt"]["checkpoint_sha256"],
                        initial_p3=initial,
                        trace=tr,
                        seconds=time.perf_counter() - start,
                        contract=fingerprint,
                    )
                    write(manifest, data)
                    print(name, round(data[name]["seconds"], 1), flush=True)
        return
    if len(data) != len(METHODS) * len(SEEDS) * 2 or not args.seed_file:
        raise ValueError("complete training bundle and private seed required")
    for name, row in data.items():
        if digest(out / (name + ".pt")) != row["checkpoint_sha256"] or row["contract"] != fingerprint:
            raise ValueError("frozen model mismatch")
    fixtures = make_relations("test", int(args.seed_file.read_text()))
    panels = [(raw, controls(raw)) for raw in fixtures.values()]
    results = {}
    for name, row in data.items():
        model = Student(row["arm"], torch.load(args.base / f"C-{row['seed']}-coordinated.pt", weights_only=True))
        model.load_state_dict(torch.load(out / (name + ".pt"), weights_only=True))
        model.select("ldt")
        rows = [score(model, raw, ref) for raw, ref in panels]
        summary = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("brier", "code_bayes", "side_only", "best_linear", "oracle")
        }
        p3 = probe(model, ctx, 3)
        p4 = probe(model, ctx, 4)
        results[name] = dict(
            p3=p3,
            p4=p4,
            ldt=summary,
            pass_all=bool(p3["pass_all"] and p4["pass_all"] and summary["brier"] < summary["side_only"]),
        )
    write(out / "results.json", dict(cells=results, private_commitment=digest(args.seed_file)))


if __name__ == "__main__":
    main()
