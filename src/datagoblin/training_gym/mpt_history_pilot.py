"""Development-only, lossy-observation propositional proof routing smoke test."""

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

DEPTHS = (2, 4, 8, 16)
STEPS = (0, 1, 4, 16, 64)
ARMS = ("reset", "history", "bayes", "typed_oracle")


def generate(seed, per_depth):
    rng = random.Random(seed)
    for depth in DEPTHS:
        for index in range(per_depth):
            z = rng.randrange(2)
            types, features = [], []
            for _ in range(depth):
                y = z if rng.random() < 0.8 else 1 - z
                x = y if rng.random() < 0.65 else 1 - y
                types.append(("P", "Q") if y == 0 else ("Q", "P"))
                features.append(x)
            yield {"root": f"{seed}:{depth}:{index}", "types": types, "features": features}


def verify(types, candidate):
    return types[candidate] == "P"


def label(types):
    return int(verify(types, 1))


def sigmoid(v):
    return 1 / (1 + math.exp(-max(-50, min(50, v))))


def feature(x, previous):
    return (1.0, 2.0 * x - 1.0, float(sum(2 * y - 1 for y in previous)))


def prob(weights, x, previous):
    return sigmoid(sum(a * b for a, b in zip(weights, feature(x, previous), strict=True)))


def rows(worlds):
    for w in worlds:
        previous = []
        for x, t in zip(w["features"], w["types"], strict=True):
            y = label(t)
            yield x, tuple(previous), y
            previous.append(y)


def fit(worlds):
    # Equal total training weight per independent root.
    data = [(x, h, y, 1 / len(w["types"])) for w in worlds for x, h, y in rows([w])]
    weights = [0.0] * 3
    checkpoints = {0: tuple(weights)}
    for step in range(1, max(STEPS) + 1):
        grad = [0.0] * 3
        for x, h, y, weight in data:
            residual = (prob(weights, x, h) - y) * weight / len(worlds)
            for j, f in enumerate(feature(x, h)):
                grad[j] += residual * f
        weights = [w - 0.15 * g for w, g in zip(weights, grad, strict=True)]
        if step in STEPS:
            checkpoints[step] = tuple(weights)
    return checkpoints


def loss(worlds, weights):
    total = 0.0
    for w in worlds:
        for x, h, y in rows([w]):
            p = prob(weights, x, h)
            total += -math.log(max(1e-15, p if y else 1 - p)) / len(w["types"])
    return total / len(worlds)


def bayes(x, history):
    # Exact generator-model posterior; a greedy first-candidate decision.
    m = sum(2 * y - 1 for y in history)
    pz = 1 / (1 + 4.0 ** (-m))
    prior = 0.2 + 0.6 * pz
    like1 = 0.65 if x else 0.35
    like0 = 1 - like1
    return prior * like1 / (prior * like1 + (1 - prior) * like0)


def rollout(w, weights, arm, budget):
    previous, attempts = [], []
    n = 0
    for i, (x, t) in enumerate(zip(w["features"], w["types"], strict=True)):
        if n >= budget:
            break
        if arm == "typed_oracle":
            a = label(t)
        elif arm == "bayes":
            a = int(bayes(x, previous) >= 0.5)
        elif arm == "reset":
            a = int(prob(weights, x, []) >= 0.5)
        elif arm == "history":
            a = int(prob(weights, x, previous) >= 0.5)
        else:
            raise ValueError(arm)
        passed = False
        for c in (a, 1 - a):
            if n >= budget:
                break
            passed = verify(t, c)
            n += 1
            attempts.append({"obligation": i, "candidate": c, "passed": passed})
            if passed:
                previous.append(c)
                break
        if not passed:
            break
    return {
        "root": w["root"], "arm": arm, "depth": len(w["types"]), "budget": budget,
        "solved": len(previous) == len(w["types"]), "checks": n, "attempts": attempts,
    }


def lean_source(worlds):
    lines = ["-- Generated development-only certificates. No target declaration in context."]
    for wi, w in enumerate(worlds):
        for i, t in enumerate(w["types"]):
            name = f"mpt_dev_{wi}_{i}"
            good = label(t)
            bad = 1 - good
            lines.extend([
                f"theorem {name} (P Q : Prop) (h0 : {t[0]}) (h1 : {t[1]}) : P := by",
                f"  fail_if_success exact h{bad}", f"  exact h{good}", f"#print axioms {name}",
            ])
    return "\n".join(lines) + "\n"


def native_replay(executable, path, expected):
    if executable is None:
        return {"status": "NOT_RUN", "reason": "no explicit Lean executable supplied"}
    exe = executable.resolve(strict=True)
    result = subprocess.run([str(exe), str(path.resolve())], capture_output=True, text=True, timeout=120, check=False)
    (path.parent / "lean.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    reported = result.stdout.count("does not depend on any axioms")
    return {
        "status": "PASS" if result.returncode == 0 and reported == expected else "FAIL",
        "returncode": result.returncode, "expected_axiom_reports": expected, "empty_axiom_reports": reported,
        "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lean", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    train = list(generate(901721, 128))
    dev = list(generate(901722, 16))
    assert not {w["root"] for w in train} & {w["root"] for w in dev}
    checkpoints = fit(train)
    results, traces = [], []
    for step, w in checkpoints.items():
        arms = {}
        for arm in ARMS:
            rr = [rollout(world, w, arm, len(world["types"]) + len(world["types"]) // 2) for world in dev]
            traces.extend(dict(r, checkpoint=step) for r in rr)
            arms[arm] = {"solved": sum(r["solved"] for r in rr), "checks": sum(r["checks"] for r in rr)}
        results.append({"step": step, "witness_nll": loss(dev, w), "weights": w, "arms": arms})
    final = [t for t in traces if t["checkpoint"] == STEPS[-1]]
    r = {t["root"]: t["solved"] for t in final if t["arm"] == "reset"}
    h = {t["root"]: t["solved"] for t in final if t["arm"] == "history"}
    paired = {
        "wins": sum(h[k] and not r[k] for k in h), "losses": sum(r[k] and not h[k] for k in h),
        "ties": sum(h[k] == r[k] for k in h),
    }
    source = lean_source(dev)
    path = args.out / "dev_certificates.lean"
    path.write_text(source, encoding="utf-8")
    expected = sum(len(w["types"]) for w in dev)
    native = native_replay(args.lean, path, expected)
    receipt = {
        "schema": "mpt-gym-01-smoke-v1", "scope": "synthetic lossy-observation development only",
        "train_roots": len(train), "dev_roots": len(dev), "dev_obligations": expected,
        "parameters": 3, "checkpoints": results, "paired_final": paired, "native_lean": native,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lean_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "elapsed_s": time.perf_counter() - start,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (args.out / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if native["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
