"""One public, post-fit local test of the three-pass clock. No parameter updates."""

import argparse
import json
from pathlib import Path

import torch
from corpus import load_corpus, rule_rows
from gym import atomic_json, file_sha
from study import dense, metrics, score, select_panel

from datagoblin.training_gym import rlt_pretrain_data


def rms(x):
    return x.float().square().mean().sqrt().item()


@torch.inference_mode()
def diagnose(model, row):
    # The miniatures use a single residual stream. Keep the same input and checkpoint for both clocks.
    x = torch.tensor([row.prompt])
    u = model.blocks[0](model.embed(x))

    def core(h):
        for block in model.blocks[1:11]:
            h = block(h)
        return h

    def residual(h):
        return core(h) - h

    d1 = residual(u)
    c, delta = 5 / 12, 1 / 12
    v = u + c * d1
    d2 = residual(v)
    equal_clock = v + c * d2
    w = u + (c - delta) * d1
    staggered = w + (c + delta) * residual(w)
    reverse_mid = u + (c + delta) * d1
    reverse = reverse_mid + (c - delta) * residual(reverse_mid)
    actual = staggered - equal_clock
    norms = {}
    for eps in (0.001, 0.003):
        jac_times_d1 = (residual(v + eps * d1) - residual(v - eps * d1)) / (2 * eps)
        # Derivative of pass-2 state at delta=0, when alpha1=c-delta and alpha2=c+delta.
        tangent = delta * (d2 - d1 - c * jac_times_d1)
        norms[str(eps)] = {"linearized_rms": rms(tangent), "remainder_rms": rms(actual - tangent)}
    direct = model(x, "direct", 3)
    local = model(x, "pass_rlt", 3)
    damp = model(x, "pass_damp", 3)
    z1, z2 = core(staggered), core(equal_clock)
    # Same-offset diagonal affine core calibrated at u. For affine F the two
    # relaxed maps commute exactly; this measures finite-precision roundoff
    # for this control only.
    def affine(z):
        return d1 + 0.125 * (z - u)

    affine_forward = w + (c + delta) * affine(w)
    affine_reverse = reverse_mid + (c - delta) * affine(reverse_mid)
    return {
        "lane": row.lane,
        "root": row.root,
        "input_rms": rms(u),
        "first_core_delta_rms": rms(d1),
        "pass2_clock_delta_rms": rms(actual),
        "pass3_clock_delta_rms": rms(z1 - z2),
        "pass2_reversed_clock_delta_rms": rms(staggered - reverse),
        "pass2_affine_reverse_roundoff_rms": rms(affine_forward - affine_reverse),
        "output_logit_clock_delta_rms": rms(local[:, -1] - damp[:, -1]),
        "output_logit_placement_delta_rms": rms(direct[:, -1] - damp[:, -1]),
        "finite_differences": norms,
    }


def run(cell, pack):
    receipt = json.loads((cell / "receipt.json").read_text())
    if file_sha(cell / "checkpoint.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    state = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = receipt["config"]
    if cfg != state["cfg"]:
        raise ValueError("conflicting checkpoint config")
    model = dense(cfg["width"])
    model.load_state_dict(state["model"])
    model.eval()
    _, dev, source = load_corpus(pack)
    if cfg["source"] != source:
        raise ValueError("different public dataset")
    samples = []
    for rows in dev.values():
        unique = {}
        for row in rows:
            unique.setdefault(row.root, row)
        samples.extend(list(unique.values())[:3])
    checks = [diagnose(model, row) for row in samples]
    ood = {}
    for split in ("length_ood", "revision_ood"):
        rows = list(rule_rows(rlt_pretrain_data.rule_rows(pack / "revision", split)))
        panel = select_panel({"rule": rows}, 32)
        for depth in (cfg["r"], 6):
            result, _ = score(model, panel, cfg["arm"], depth, 1)
            ood[f"{split}@{depth}"] = {"metrics": metrics(result), "rows": result}
    return {
        "scope": "post-fit public diagnostics on frozen weights",
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "three_pass_perturbation": checks,
        "ood": ood,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    atomic_json(a.out, run(a.cell, a.pack))
