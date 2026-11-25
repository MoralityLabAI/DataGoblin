"""ALEPH-04. Exact local channels, native LDT selection and hard-message learners."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time
from collections import Counter
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

PLACEMENT = Path(__file__).resolve().parents[1] / "aleph_rlt_placement"
sys.path.insert(0, str(PLACEMENT))
from receipt_ldt import TOLERANCE, _candidate, _label  # noqa: E402

from datagoblin.experiments.ldt_array_hard_surface import derive_array_target  # noqa: E402

SEEDS = (137, 163, 191)
ARMS = ("C", "D", "MLP")
BUDGETS = (1, 2, 4, 8)
SITE_MASS = np.array([9, 5, 20, 7, 16, 6, 13, 24]) / 100
WORLDS = (np.arange(8)[:, None] >> np.arange(3)) & 1
CODES = (2 * np.arange(128)[:, None] >> np.arange(8)) & 1
UNIT = 8 * math.lcm(*range(1, 9))


def canonical(truth):
    truth = np.asarray(truth, dtype=np.int64)
    return int((truth ^ truth[0]) @ (1 << np.arange(8))) // 2


def shortcut_codes(targets):
    linear = {canonical((WORLDS @ np.array(w)) % 2) for w in product(range(2), repeat=3)}
    thresholds = set()
    for weights in product(range(-3, 4), repeat=3):
        z = WORLDS @ np.array(weights)
        thresholds.update(canonical(z > t) for t in np.unique(z))
    # 104 distinct affine Boolean threshold functions on three variables include complements.
    assert len(thresholds) == 52
    return linear, thresholds | linear | {canonical(q) for q in targets.T}


def risk_units(partitions, labels, side):
    """Each query's exact Bayes Brier numerator at denominator 8*lcm(1,...,8)."""
    out = np.zeros((len(partitions), labels.shape[1]), dtype=np.int64)
    for message in range(int(partitions.max()) + 1):
        for verified in (0, 1):
            cell = (partitions[:, :, None] == message) & (side == verified)
            count = cell.sum(1)
            yes = (cell * labels).sum(1)
            out += yes * (count - yes) * (UNIT // 8 // np.maximum(1, count))
    return out


def local_program(site, trial):
    rng = np.random.default_rng(71923 + 7919 * site + 2347 * trial)
    family = "conditional_obligation" if site % 2 == 0 else "native_two_rival"
    features, labels, sides = [], [], []
    for q, (i, j, k) in enumerate(permutations(range(3))):
        flip = rng.integers(0, 2, 3)
        signed = 1 - 2 * flip
        bits = WORLDS ^ flip
        row = np.zeros(12)
        if family == "conditional_obligation":
            row[i], row[3 + j], row[3 + k] = signed[i], signed[j], 2 * signed[k]
            row[9] = 1
            answers = np.where(bits[:, i], bits[:, k], bits[:, j])
        else:
            first, second = np.zeros((2, 3), bool)
            first[[i, j]] = True
            uncertain = q % 2 == 1
            second[[k] if uncertain else [j, k]] = True
            row[:3], row[3:6] = signed * first, signed * second
            row[10:] = (1, int(uncertain))
            native = "array_unresolved_after_challenge" if uncertain else "array_two_opponents"
            lo, hi, width = np.full(3, 0.25), np.full(3, 0.75), np.full(3, 0.04)
            answers = np.array([_label(native, np.where(x, hi, lo), lo, hi, first, second, width) for x in bits])
            assert np.array_equal(answers, bits[:, first].any(1) & bits[:, second].any(1))
        for axis in range(3):
            features.append(row.copy())
            features[-1][6 + axis] = 1
            labels.append(answers)
            sides.append(WORLDS[:, axis])
    return family, np.asarray(features), np.array(labels, dtype=int).T, np.array(sides).T


def central_partitions():
    """Restricted-growth strings enumerate each at-most-four-message partition once."""
    result = []

    def grow(prefix):
        if len(prefix) == 8:
            result.append(prefix)
            return
        for label in range(min(3, max(prefix) + 1) + 1):
            grow([*prefix, label])

    grow([0])
    return np.asarray(result)


def objectives(site, trial, unit_risk, shortcuts, rejected):
    rng = np.random.default_rng(123991 + 1117 * site + trial)
    accepted = []
    for _ in range(4000):
        w = rng.integers(1, 21, unit_risk.shape[1]) ** 2
        endpoints = np.stack([64 * w, 63 * w + np.roll(w, 1)])
        risks = unit_risk @ endpoints.T
        best = int(risks[:, 0].argmin())
        remaining = np.delete(risks, best, axis=0)
        if np.any(remaining.min(0) <= risks[best]):
            rejected["unstable_or_tied"] += 1
            continue
        if best in shortcuts:
            rejected["shortcut"] += 1
            continue
        if any(best == x["code"] for x in accepted):
            rejected["unchanged_objective_code"] += 1
            continue
        margin = risks[sorted(shortcuts)].min(0) - risks[best]
        if np.any(1000 * margin < UNIT * endpoints.sum(1)):
            rejected["shortcut_margin_below_0.001"] += 1
            continue
        accepted.append(dict(code=best, endpoints=endpoints, margin=margin))
        if len(accepted) == 2:
            break
    return accepted


def build_programs(start_trial=0):
    result = []
    for site in range(8):
        failures = Counter()
        for trial in range(start_trial, start_trial + 16):
            family, features, labels, side = local_program(site, trial)
            risks = risk_units(CODES, labels, side)
            linear, shortcuts = shortcut_codes(labels)
            profiles = objectives(site, trial, risks, shortcuts, failures)
            if len(profiles) == 2:
                result.append(
                    dict(
                        site=site,
                        trial=trial,
                        family=family,
                        features=features,
                        y=labels,
                        side=side,
                        risks=risks,
                        linear=sorted(linear),
                        profiles=profiles,
                        rejections=dict(failures),
                    )
                )
                break
            failures["program_with_insufficient_objectives"] += 1
        else:
            raise RuntimeError(f"no certified local program for {site}: {failures}")
    return result


def mix(profile, t):
    w = (1 - t) * profile["endpoints"][0] + t * profile["endpoints"][1]
    return w / w.sum()


def certificate(programs):
    four = central_partitions()
    assert len(four) == 2795
    linear = np.asarray([WORLDS @ v % 2 for v in product(range(2), repeat=3)])
    rank_two = np.asarray([a + 2 * b for a, b in combinations(linear, 2)])
    rows = []
    for g in programs:
        central = risk_units(four, g["y"], g["side"])
        lin2 = risk_units(rank_two, g["y"], g["side"])
        profiles = []
        for p in g["profiles"]:
            ep = p["endpoints"]
            den = UNIT * ep.sum(1)
            profiles.append(
                dict(
                    code_mask=2 * p["code"],
                    endpoints=ep.tolist(),
                    min_shortcut_margin=float(min(p["margin"] / den)),
                    integer_optimal_numerators=(g["risks"][p["code"]] @ ep.T).tolist(),
                    denominators=den.tolist(),
                    central_midpoint_brier={
                        "1": float(min(g["risks"] @ mix(p, 0.5)) / UNIT),
                        "2": float(min(central @ mix(p, 0.5)) / UNIT),
                        "4": 0.0,
                        "8": 0.0,
                    },
                    linear_midpoint_brier={
                        "1": float(min(g["risks"][g["linear"]] @ mix(p, 0.5)) / UNIT),
                        "2": float(min(lin2 @ mix(p, 0.5)) / UNIT),
                        "4": 0.0,
                        "8": 0.0,
                    },
                )
            )
        rows.append(
            dict(
                site=g["site"],
                family=g["family"],
                trial=g["trial"],
                profiles=profiles,
                rejected=g["rejections"],
                linear_code_count=len(g["linear"]),
            )
        )
    return dict(
        schema="aleph04-certified-programs-v1",
        side="independent verified coordinate",
        local_partitions_1bit=128,
        local_partitions_2bit=len(four),
        oracle_unit=UNIT,
        rows=rows,
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def public_inputs(g, objective):
    sender = np.zeros((8, 12))
    sender[:, :3] = 2 * WORLDS - 1
    sender[:, 3 + g["site"]] = 1
    sender[:, 11] = 2 * objective - 1
    receiver = g["features"].copy()
    # The first nine columns fully specify the public predicate masks and side axis.
    # The site identifies the grammar. Interval LDT rivals have singleton second masks.
    receiver[:, 9] = (g["site"] - 3.5) / 3.5
    receiver[:, 10] = 2 * objective - 1
    receiver[:, 11] = 1  # Visibility of the public channel; zero in matched no-message calls.
    return sender, receiver


def posterior(code, labels, side):
    target = np.zeros(labels.shape, dtype=float)
    for m, s in product((0, 1), repeat=2):
        hits = (code[:, None] == m) & (side == s)
        prob = (hits * labels).sum(0) / np.maximum(hits.sum(0), 1)
        target += hits * prob
    return target


def training_arrays(programs):
    samples, probs = [], []
    for g in programs:
        for objective, profile in enumerate(g["profiles"]):
            p, q = public_inputs(g, objective)
            code = CODES[profile["code"]]
            valid = posterior(code, g["y"], g["side"])
            blind = posterior(np.zeros(8), g["y"], g["side"])
            nw, nq = g["y"].shape
            samples.append(
                (
                    np.repeat(p, nq, 0),
                    np.tile(q, (nw, 1)),
                    g["side"].flatten(),
                    np.repeat(code, nq),
                    valid.flatten(),
                    blind.flatten(),
                )
            )
            # Sampling the two endpoints equally is exactly their mixture for this loss.
            probs.extend(np.tile(mix(profile, 0.5), nw) / (8 * 2 * nw))
    return [np.concatenate(a) for a in zip(*samples, strict=True)], np.array(probs)


def source_hashes():
    paths = [
        Path(__file__),
        Path(__file__).with_name("NONLINEAR_PROTOCOL.md"),
        PLACEMENT / "receipt_models.py",
        PLACEMENT / "receipt_ldt.py",
        PLACEMENT.parents[1] / "src/datagoblin/experiments/ldt_array_hard_surface.py",
    ]
    return {p.name: digest(p) for p in paths}


def cpu_torch():
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def train(programs, out):
    torch = cpu_torch()
    from receipt_models import ReceiptModel

    if (out / "training.json").exists() or any(out.glob("*.pt")):
        raise ValueError("training requires an empty checkpoint directory")
    arrays, prob = training_arrays(programs)
    p, q, side, message, valid, blind = [torch.as_tensor(a, dtype=torch.float32) for a in arrays]
    q0 = q.clone()
    q0[:, 11] = 0
    runs = {}
    steps, batch = 640, 64
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        stream = rng.choice(len(p), size=(steps, batch), p=prob)
        stream_hash = hashlib.sha256(stream.astype("<i8").tobytes()).hexdigest()
        for arm in ARMS:
            torch.manual_seed(seed)
            start = time.monotonic()
            model = ReceiptModel(arm)
            opt = torch.optim.Adam(model.parameters(), lr=0.004)
            trace = []
            for step, index in enumerate(stream):
                idx = torch.as_tensor(index)
                logits = model.encode(p[idx])
                m, s = message[idx], side[idx]
                yes = model.decode(q[idx], s, m).sigmoid()
                no = model.decode(q0[idx], s, torch.zeros_like(m)).sigmoid()
                encoder = torch.nn.functional.binary_cross_entropy_with_logits(logits, m)
                readout = ((yes - valid[idx]).square() + (no - blind[idx]).square()).mean()
                loss = encoder + 3 * readout
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                if step % 100 == 0 or step + 1 == steps:
                    trace.append([step + 1, float(encoder.detach()), float(readout.detach())])
            path = out / f"{arm}-{seed}.pt"
            torch.save(model.state_dict(), path)
            runs[f"{arm}-{seed}"] = dict(
                architecture=arm,
                seed=seed,
                params=sum(x.numel() for x in model.parameters()),
                logical_visits=2 if arm != "MLP" else 1,
                steps=steps,
                batch=batch,
                seconds=time.monotonic() - start,
                stream_sha256=stream_hash,
                checkpoint_sha256=digest(path),
                trace=trace,
            )
            print(arm, seed, round(runs[f"{arm}-{seed}"]["seconds"], 2), flush=True)
    dump(
        out / "training.json",
        dict(
            schema="aleph04-training-v1",
            runs=runs,
            sources=source_hashes(),
            optimizer="Adam lr .004, clipped norm 1; full-code BCE + 3*(valid and no-message posterior Brier)",
            certificate=digest(out / "certificate.json"),
            transfer_certificate=digest(out / "transfer-certificate.json"),
            torch=str(torch.__version__),
        ),
    )


def native_selection(gains, budget):
    """Verified gain under the two announced endpoint lotteries forms a native LDT record."""
    scale = max(float(gains.max()), 1e-12)
    candidates = [_candidate(np.array([a, b, min(a, b)]) / scale) for a, b in gains]
    remaining = list(range(len(candidates)))
    selected = []
    for _ in range(budget):
        report = derive_array_target([candidates[i] for i in remaining], tolerance=TOLERANCE)
        frontier = report["selected_indices"]
        # Fixed public site-ID tie order, before any local world or side receipt is drawn.
        chosen = min(remaining[i] for i in frontier)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def actual_site(model_from, model_to, g, o, w, torch):
    p, q = public_inputs(g, o)
    p, q = [torch.as_tensor(a, dtype=torch.float32) for a in (p, q)]
    with torch.no_grad():
        code = (model_from.encode(p) >= 0).float()
        q = q.repeat(8, 1)
        side = torch.as_tensor(g["side"].flatten(), dtype=torch.float32)
        yes = model_to.decode(q, side, code.repeat_interleave(len(w))).sigmoid().reshape(8, -1).numpy()
        q0 = q.clone()
        q0[:, 11] = 0
        no = model_to.decode(q0, side, torch.zeros_like(side)).sigmoid().reshape(8, -1).numpy()
    mask = canonical(code.numpy())
    return dict(
        code_mask=2 * mask,
        valid=float(((yes - g["y"]) ** 2).mean(0) @ w),
        no_message=float(((no - g["y"]) ** 2).mean(0) @ w),
        ideal=float(g["risks"][mask] @ w / UNIT),
        optimal=bool(mask == g["profiles"][o]["code"]),
    )


def integrate(base, active, chosen):
    mask = np.zeros(8, bool)
    mask[chosen] = True
    return float(SITE_MASS @ np.where(mask, active, base))


def design(programs, objective, fractions):
    side, oracle, linear, unaware, endpoint_gains = [], [], [], [], []
    for g, t in zip(programs, fractions, strict=True):
        p = g["profiles"][objective]
        w = mix(p, t)
        risks = g["risks"] @ w / UNIT
        side.append(risks[0])
        oracle.append(risks[p["code"]])
        linear.append(min(risks[g["linear"]]))
        no_objective = (mix(g["profiles"][0], 0.5) + mix(g["profiles"][1], 0.5)) / 2
        blind_code = int((g["risks"] @ no_objective).argmin())
        unaware.append(risks[blind_code])
        ep = p["endpoints"]
        ep = ep / ep.sum(1, keepdims=True)
        endpoint_gains.append((g["risks"][0] - g["risks"][p["code"]]) @ ep.T / UNIT)
    side, oracle, linear, unaware = map(np.array, (side, oracle, linear, unaware))
    data = {}
    for b in BUDGETS:
        chosen = native_selection(SITE_MASS[:, None] * endpoint_gains, b)
        top = np.argsort(-(side - oracle) * SITE_MASS)[:b]
        top_linear = np.argsort(-(side - linear) * SITE_MASS)[:b]
        exact = integrate(side, oracle, top)
        base = float(SITE_MASS @ side)
        data[b] = dict(
            authorized_sites=chosen,
            oracle_sites=top.tolist(),
            side_only=base,
            nonlinear_optimum=exact,
            linear_optimum=integrate(side, linear, top_linear),
            native_ldt=integrate(side, oracle, chosen),
            uniform_schedule=base - float(SITE_MASS @ (side - oracle)) * b / 8,
            round_robin=integrate(side, oracle, list(range(b))),
            objective_blind=integrate(side, unaware, chosen),
            oracle_utility_per_bit=(base - exact) / b,
        )
    return data


def score_transfer(programs, models, fractions, torch):
    rows = []
    refs = [design(programs, o, frac) for o, frac in enumerate(fractions)]
    for seed, o in product(SEEDS, range(2)):
        for arm in ARMS:
            cells = [
                actual_site(models[arm, seed], models[arm, seed], g, o, mix(g["profiles"][o], t), torch)
                for g, t in zip(programs, fractions[o], strict=True)
            ]
            no = np.array([c["no_message"] for c in cells])
            yes = np.array([c["valid"] for c in cells])
            for b, ref in refs[o].items():
                loss = integrate(no, yes, ref["authorized_sites"])
                rows.append(
                    dict(
                        seed=seed,
                        objective=o,
                        arm=arm,
                        bits=b,
                        brier=loss,
                        matched_no_message=float(SITE_MASS @ no),
                        utility_per_bit=float(SITE_MASS @ no - loss) / b,
                        optimal_senders=sum(c["optimal"] for c in cells),
                    )
                )
    return dict(panels=refs, measured=rows)


def score(programs, transfer_candidates, out):
    torch = cpu_torch()
    from receipt_models import ReceiptModel

    manifest = json.loads((out / "training.json").read_bytes())
    if (
        manifest["sources"] != source_hashes()
        or manifest["certificate"] != digest(out / "certificate.json")
        or manifest["transfer_certificate"] != digest(out / "transfer-certificate.json")
    ):
        raise ValueError("source or certificate hash mismatch")
    models = {}
    for arm, seed in product(ARMS, SEEDS):
        name = f"{arm}-{seed}"
        path = out / f"{name}.pt"
        if digest(path) != manifest["runs"][name]["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch for {name}")
        models[arm, seed] = ReceiptModel(arm).eval()
        models[arm, seed].load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    private = out / "private-eval.seed"
    if not private.exists():
        with private.open("x", encoding="ascii") as file:
            file.write(str(secrets.randbits(128)))
    rng = np.random.default_rng(int(private.read_text(encoding="ascii")))
    fractions = rng.integers(1, 65536, size=(2, 8)) / 65536
    transfer_index = int(rng.integers(len(transfer_candidates)))
    fresh_fractions = rng.integers(1, 65536, size=(2, 8)) / 65536
    panels = [design(programs, o, frac) for o, frac in enumerate(fractions)]
    measured, sites = [], []
    for seed in SEEDS:
        for o, frac in enumerate(fractions):
            saved = {}
            for sender, reader in product(ARMS, repeat=2):
                cells = [
                    actual_site(models[sender, seed], models[reader, seed], g, o, mix(g["profiles"][o], t), torch)
                    for g, t in zip(programs, frac, strict=True)
                ]
                saved[sender, reader] = cells
                sites.append(dict(seed=seed, objective=o, sender=sender, receiver=reader, cells=cells))
            pairs = [(f"{s}->{r}", v) for (s, r), v in saved.items()]
            cross = [saved[ARMS[i % 3], ARMS[(i + 1) % 3]][i] for i in range(8)]
            pairs.append(("heterogeneous", cross))
            for label, cells in pairs:
                no = np.array([c["no_message"] for c in cells])
                yes = np.array([c["valid"] for c in cells])
                ideal = np.array([c["ideal"] for c in cells])
                base = float(SITE_MASS @ no)
                exact_side = np.array(
                    [g["risks"][0] @ mix(g["profiles"][o], t) / UNIT for g, t in zip(programs, frac, strict=True)]
                )
                for b, ref in panels[o].items():
                    real = integrate(no, yes, ref["authorized_sites"])
                    attainable = integrate(exact_side, ideal, ref["authorized_sites"])
                    measured.append(
                        dict(
                            seed=seed,
                            objective=o,
                            pair=label,
                            bits=b,
                            brier=real,
                            matched_no_message=base,
                            ideal_for_learned_bits=attainable,
                            utility_per_bit=(base - real) / b,
                            increment_vs_ideal_side=ref["side_only"] - real,
                            optimal_senders=sum(c["optimal"] for c in cells),
                        )
                    )
    gate = {}
    for arm in ARMS:
        passes = []
        for seed in SEEDS:
            rows = [r for r in measured if r["seed"] == seed and r["pair"] == f"{arm}->{arm}" and r["bits"] == 4]
            passes.append(np.mean([r["brier"] - panels[r["objective"]][4]["linear_optimum"] for r in rows]) < 0)
        gate[arm] = [bool(v) for v in passes]
    transfer = score_transfer(transfer_candidates[transfer_index], models, fresh_fractions, torch)
    dump(
        out / "results.json",
        dict(
            schema="aleph04-population-v1",
            private_seed_sha256=digest(private),
            training_sha256=digest(out / "training.json"),
            sources=source_hashes(),
            panels=panels,
            site_masses=SITE_MASS.tolist(),
            local_measurements=sites,
            measured=measured,
            gate_pass_by_seed=gate,
            fresh_program_index=transfer_index,
            fresh_program_transfer=transfer,
            heterogeneous_roles=[dict(site=i, sender=ARMS[i % 3], receiver=ARMS[(i + 1) % 3]) for i in range(8)],
        ),
    )
    print("gate", gate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("certify", "train", "score"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    programs = build_programs()
    transfers = [build_programs(offset) for offset in (100, 200)]
    cert_path = args.out / "certificate.json"
    transfer_path = args.out / "transfer-certificate.json"
    if args.phase == "certify":
        dump(cert_path, certificate(programs))
        dump(transfer_path, [certificate(group) for group in transfers])
        print("certified", len(programs), "worker pairs", digest(cert_path), flush=True)
        return
    if json.loads(cert_path.read_bytes()) != certificate(programs):
        raise ValueError("certificate changed before training or scoring")
    if json.loads(transfer_path.read_bytes()) != [certificate(group) for group in transfers]:
        raise ValueError("transfer certificate changed before training or scoring")
    if args.phase == "train":
        train(programs, args.out)
    else:
        score(programs, transfers, args.out)


if __name__ == "__main__":
    main()
