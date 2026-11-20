"""Train and score matched feedback policies on the original Hermes repair block."""

from __future__ import annotations

from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_text
from datagoblin.serialization import load_structured

from . import rlt_hermes_repair as protocol
from .rlt_lean_labels import atomic_record


def visits(model, x, valid, mode, depth, inner):
    import torch

    from .rlt_pilot import feedback_weight

    pad = valid < 0.5
    encoded = model.inp(x) + model.row(model.r_idx) + model.col(model.c_idx)
    y = torch.zeros_like(encoded)
    z = torch.zeros_like(encoded)
    for index in range(depth):
        for _ in range(inner):
            z = model.block(encoded + y + z, src_key_padding_mask=pad)
        endpoint = model.block(y + z, src_key_padding_mask=pad)
        weight = feedback_weight(mode, index, depth)
        update = endpoint - y
        y = endpoint if weight == 1.0 else y + weight * update
        yield y, weight * update


def menu(campsite, task, gold, grid):
    checks = int(grid is not None)
    if checks and campsite.passes(task, grid):
        return "original", checks, 0
    for attempts, name in enumerate(("c_repair", "dual_repair"), 1):
        candidate = campsite.repair(task, gold, grid, name)
        if candidate is not None:
            checks += 1
            if campsite.passes(task, candidate):
                return name, checks, attempts
    return None, checks, 2


def fresh_grids(root, prompts, tasks, campsite):
    receipt = root / "rollout-receipt.json"
    if not receipt.exists():
        return {}
    rollout = load_structured(receipt)
    if (rollout["protocol_sha256"] != sha256_file(root / "rollout-protocol.json")
            or rollout["prepared_sha256"] != sha256_file(root / "prepared.json")
            or rollout["count"] != len(prompts) or rollout["rollouts"].keys() != prompts.keys()):
        raise ValueError("rollout receipt does not match the prepared test cohort")
    fresh = {}
    for pid, prompt in prompts.items():
        path = root / "rollouts" / f"{pid}.json"
        row = load_structured(path)
        if (rollout["rollouts"][pid] != sha256_file(path) or row["id"] != pid
                or row["prompt_sha256"] != sha256_text(prompt)):
            raise ValueError("rollout record mismatch")
        fresh[pid] = campsite.parse_grid(row["completion"], len(tasks[pid][0].grid))
    return fresh


def cohorts(root, inputs, campsite, cells, guard):
    tasks = campsite.task_by_id("test")
    fresh = fresh_grids(root, inputs["fresh"], tasks, campsite)
    result = {}
    for name, candidates in (("archival", inputs["archival"]), ("fresh", fresh)):
        if not candidates:
            continue
        rows = []
        for pid, grid in candidates.items():
            guard()
            task, gold = tasks[pid]
            operator, checks, attempts = menu(campsite, task, gold, grid)
            baseline = {"id": pid, "size": len(task.grid), "menu_operator": operator,
                        "menu_verifier_calls": checks, "projection_attempts": attempts,
                        "parsed": grid is not None, "rectangular": campsite.rectangular(task, grid)}
            cand = cells.normalise(task.grid, grid)
            x, _, _, valid = cells.encode(protocol.FRAMING, task.grid, task.row_constraints, task.col_constraints, cand)
            rows.append((task, x, valid, baseline))
        result[name] = rows
    if sum(row["menu_operator"] is not None for *_, row in result["archival"]) != 63:
        raise ValueError("the pinned upstream menu did not reproduce the paper's 63/148 archive score")
    return result


def summarize(rows, depth, inner):
    first = [next((i for i, ok in enumerate(row["cycle_passes"], 1) if ok), None) for row in rows]
    menu_calls = sum(row["menu_verifier_calls"] for row in rows)
    pending = [
        (row["cycle_passes"][-1], certified_at)
        for row, certified_at in zip(rows, first, strict=True)
        if row["menu_operator"] is None
    ]
    menu_solved = len(rows) - len(pending)
    stop_cycles = sum(certified_at or depth for _, certified_at in pending)
    return {
        "cases": len(rows), "parsed": sum(row["parsed"] for row in rows),
        "rectangular": sum(row["rectangular"] for row in rows),
        "raw_valid": sum(row["menu_operator"] == "original" for row in rows), "menu_solved": menu_solved,
        "projection_attempts": sum(row["projection_attempts"] for row in rows),
        "menu_verifier_calls": menu_calls,
        "standalone_final": sum(row["cycle_passes"][-1] for row in rows),
        "fixed_solved": menu_solved + sum(ok for ok, _ in pending),
        "stop_solved": menu_solved + sum(certified_at is not None for _, certified_at in pending),
        "first_valid_cycle_on_open_cases": [sum(first_cycle == i for _, first_cycle in pending)
                                            for i in range(1, depth + 1)],
        "fixed_policy_verifier_calls": menu_calls + len(pending),
        "stop_policy_verifier_calls": menu_calls + stop_cycles,
        "fixed_policy_block_calls": (inner + 1) * depth * len(pending),
        "stop_policy_block_calls": (inner + 1) * stop_cycles,
        "diagnostic_replay_verifier_calls": menu_calls + depth * len(rows),
        "diagnostic_replay_block_calls": (inner + 1) * depth * len(rows),
        "invalid_commits": 0,
    }


def train(checkpoint, trm, tensors, mode, seed, batch_plan, guard):
    import torch
    import torch.nn.functional as functional

    features, labels, masks, valid = tensors
    torch.manual_seed(seed)
    model = trm.build(features.shape[-1]).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol.STEPS)
    losses = []
    step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=features.device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        losses, step = saved["losses"], saved["step"]
    with torch.inference_mode():
        model.eval()
        x, v = features[:2], valid[:2]
        direct = list(visits(model, x, v, "direct", trm.CYCLES, trm.INNER))[-1][0]
        torch.testing.assert_close(model.head(direct).squeeze(-1), model(x, v), rtol=0, atol=0)
    model.train()
    for index in range(step, protocol.STEPS):
        guard()
        batch = batch_plan[index]
        for state, _ in visits(model, features[batch], valid[batch], mode, trm.CYCLES, trm.INNER):
            hidden = state
        logits = model.head(hidden).squeeze(-1)
        loss = (functional.binary_cross_entropy_with_logits(logits, labels[batch], reduction="none")
                * masks[batch]).sum() / masks[batch].sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (index + 1) % 40 == 0 or index + 1 == protocol.STEPS:
            pending = checkpoint.with_suffix(".part")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": index + 1, "losses": losses}, pending)
            pending.replace(checkpoint)
        if (index + 1) % 160 == 0:
            print(f"{mode} seed {seed} update {index + 1}/{protocol.STEPS} loss {losses[-1]:.4f}", flush=True)
    return model.eval(), losses


def assess(model, mode, trm, campsite, cells, records, guard):
    import numpy as np
    import torch

    rows = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            guard()
            batch = records[start:start + 32]
            x = torch.as_tensor(np.asarray([item[1] for item in batch]), dtype=torch.float32, device=device)
            valid = torch.as_tensor(np.asarray([item[2] for item in batch]), dtype=torch.float32, device=device)
            scores = []
            for y, update in visits(model, x, valid, mode, trm.CYCLES, trm.INNER):
                logits = model.head(y).squeeze(-1).tolist()
                magnitudes = [
                    ((state.square().mean(dim=-1) * valid).sum(dim=-1) / valid.sum(dim=-1)).sqrt().tolist()
                    for state in (y, update)
                ]
                scores.append((logits, *magnitudes))
            for i, (task, _, _, base) in enumerate(batch):
                rows.append({
                    **base, "cycle_passes": [campsite.passes(task, cells.decode(task.grid, logits[i]))
                                             for logits, _, _ in scores],
                    "state_rms": [norm[i] for _, norm, _ in scores],
                    "update_rms": [norm[i] for _, _, norm in scores],
                })
    return {"summary": summarize(rows, trm.CYCLES, trm.INNER), "cases": rows}


def compare(root, study, hermes_lite, device, guard, initialize_device):
    import numpy as np
    import torch

    prepared, inputs = protocol.checked_inputs(root)
    common, campsite, cells, trm = protocol.original(study, hermes_lite)
    if prepared["registration"]["source"] != protocol.source_receipt(study, common, campsite, cells):
        raise ValueError("upstream curriculum changed after preparation")
    tests = cohorts(root, inputs, campsite, cells, guard)
    print(f"Reproduced the 63/148 archive menu result and prepared {sum(map(len, tests.values()))} cases", flush=True)
    initialize_device()
    registration = {
        "prepared_sha256": sha256_file(root / "prepared.json"),
        "fit_code_sha256": sha256_file(Path(__file__)),
        "cycles": trm.CYCLES, "inner": trm.INNER, "steps": protocol.STEPS,
        "batch": protocol.BATCH, "modes": list(protocol.MODES), "seeds": list(protocol.SEEDS), "fit_device": device,
        "torch_version": str(torch.__version__),
    }
    manifest = root / "fit-protocol.json"
    if manifest.exists() and load_structured(manifest) != registration:
        raise ValueError("fit protocol changed")
    atomic_record(manifest, registration)
    with np.load(root / "training.npz", allow_pickle=False) as arrays:
        tensors = tuple(torch.from_numpy(arrays[name]).to(device) for name in
                        ("features", "labels", "loss_mask", "valid"))
    output = root / "fits"
    output.mkdir(exist_ok=True)
    fits = []
    for seed in protocol.SEEDS:
        rng = np.random.default_rng(seed)
        batch_plan = torch.from_numpy(np.concatenate([
            rng.permutation(len(tensors[0])).reshape(-1, protocol.BATCH)
            for _ in range(protocol.STEPS * protocol.BATCH // len(tensors[0]))
        ])).to(device)
        for mode in protocol.MODES:
            guard()
            name = f"{mode}-{seed}"
            checkpoint = output / f"{name}.pt"
            record = output / f"{name}.json"
            if record.exists():
                result = load_structured(record)
                if result["checkpoint_sha256"] != sha256_file(checkpoint):
                    raise ValueError("fit checkpoint changed")
            else:
                result = None
            outstanding = tests.keys() - (result["cohorts"].keys() if result else set())
            if outstanding:
                model, losses = train(checkpoint, trm, tensors, mode, seed, batch_plan, guard)
                if result is None:
                    result = {"mode": mode, "seed": seed, "checkpoint_sha256": sha256_file(checkpoint),
                              "training_loss": losses, "cohorts": {}}
                for cohort in sorted(outstanding):
                    result["cohorts"][cohort] = assess(model, mode, trm, campsite, cells, tests[cohort], guard)
                atomic_record(record, result)
                del model
                print(f"Verified {name} on {', '.join(sorted(outstanding))}", flush=True)
            fits.append(result)
    full = "fresh" in tests
    rollout_sha = sha256_file(root / "rollout-receipt.json") if full else None
    evidence = root / ("evidence.json" if full else "archive-evidence.json")
    atomic_record(evidence, {"protocol": registration, "source": prepared["registration"]["source"],
                             "verified_targets": prepared["training_targets_verified"],
                             "rollout_sha256": rollout_sha, "fits": fits})
