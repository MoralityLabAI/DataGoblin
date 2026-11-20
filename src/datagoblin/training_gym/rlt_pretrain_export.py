"""Bind a completed mixed pilot and its Lean evidence into a replay package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import load_structured, pretty_json

from . import rlt_pretrain_data as data
from .rlt_lean_pool_export import HARD_CAP
from .rlt_pretrain_scale import Layout
from .rlt_pretrain_scale import report as layout_report


def resources(root, phases):
    history = list(data.json_lines(root / "resource-history.jsonl"))
    if any(r["hard_commit_cap_bytes"] != HARD_CAP or r["job_peak_commit_bytes"] > HARD_CAP for r in history):
        raise ValueError("the recorded job exceeds the experiment cap")
    terminal = {r["phase"]: r["status"] for r in history}
    if any(terminal.get(phase) != "complete" for phase in phases):
        raise ValueError("required experiment phases have not completed")
    return {"peak_job_commit_bytes": max(r["job_peak_commit_bytes"] for r in history),
            "phase_wall_s": {phase: sum(r["wall_s"] for r in history if r["phase"] == phase) for phase in phases}}


def clip_tree(root, prefix, names):
    return {f"{prefix}/{name}": root / name for name in names}


def zip_package(archive, entries):
    manifest = {
        "schema": "datagoblin.1b-pretrain-evidence-bundle.v1",
        "files": {name: sha256_file(path) for name, path in sorted(entries.items())},
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix(".part.zip")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("bundle_manifest.json", pretty_json(manifest))
        for name, path in sorted(entries.items()):
            handle.write(path, name)
    with zipfile.ZipFile(pending) as handle:
        if handle.testzip() is not None or len(handle.namelist()) != len(entries) + 1:
            raise ValueError("the written evidence bundle is corrupt")
        for name, digest in manifest["files"].items():
            measured = hashlib.sha256()
            with handle.open(name) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    measured.update(chunk)
            if measured.hexdigest() != digest:
                raise ValueError(f"packaged member differs from source: {name}")
    pending.replace(archive)
    return manifest


def audit_trace(detail, text, proofs, graphs, protocol):
    depth = detail["depth"]
    seen = set()
    corpora = defaultdict(Counter)
    for row in detail["text_windows"]:
        identity = row["id"]
        source = text[identity]
        if (
            identity in seen or row["corpus"] != source["corpus"] or row["tokens"] != len(source["tokens"]) - 1
            or not math.isfinite(row["nll"]) or row["nll"] < 0 or not 0 <= row["correct"] <= row["tokens"]
        ):
            raise ValueError("invalid independent text-window score")
        seen.add(identity)
        metric = corpora[row["corpus"]]
        metric.update(nll=row["nll"], tokens=row["tokens"], correct=row["correct"], windows=1, cell_calls=2 * depth)
    if seen != text.keys() or corpora.keys() != detail["summary"]["text"].keys():
        raise ValueError("incomplete text-window coverage")
    for name, computed in corpora.items():
        observed = detail["summary"]["text"][name]
        if any(not math.isclose(value, observed[key], abs_tol=1e-10) for key, value in computed.items()):
            raise ValueError("aggregate text scores differ from window traces")
        if not math.isclose(observed["bits_per_byte"], computed["nll"] / (computed["tokens"] * math.log(2))):
            raise ValueError("invalid bits-per-byte normalization")

    seen.clear()
    proof_tally = Counter(dict.fromkeys((
        "goals", "proved", "visits", "verifier_replay_checks", "projected_tactic_requests",
        "projected_kernel_requests", "unknown_attempts",
    ), 0))
    for row in detail["lean"]:
        public, receipt = proofs[row["id"]]
        allowed = set(public["orders"]["compose"])
        if row["id"] in seen or row["lineage"] != public["lineage"]:
            raise ValueError("duplicate Lean goal or wrong structural lineage")
        seen.add(row["id"])
        attempts = row["attempts"]
        tried = {a["id"] for a in attempts}
        limit = min(len(allowed), depth, protocol["lean_budget"])
        if not tried <= allowed or len(tried) != len(attempts) or row["visits"] != len(tried) or len(tried) > limit:
            raise ValueError("unexpected Lean proposal work")
        verified = []
        for a in attempts:
            outcome = receipt["outcomes"][a["id"]]
            if (a["label"], a["original_kernel_requests"], a["original_tactic_requests"],
                a["original_tactic_seconds"], a["original_kernel_seconds"]) != (
                outcome["label"], outcome["kernel_calls"], 1, outcome["tactic_wall_s"], outcome["kernel_wall_s"],
            ):
                raise ValueError("proof replay differs from independent checks")
            if outcome["label"] == 1:
                verified.append(a["id"])
            proof_tally.update(projected_tactic_requests=1, verifier_replay_checks=1,
                               projected_kernel_requests=outcome["kernel_calls"],
                               unknown_attempts=int(outcome["label"] is None))
        proof = row["proved"]
        if proof is None:
            if verified or len(tried) != limit:
                raise ValueError("Lean replay omitted a verified stopping opportunity")
        elif verified != [proof] or attempts[-1]["id"] != proof:
            raise ValueError("Lean replay stopped without a current verified proof")
        proof_tally.update(goals=1, proved=int(proof is not None), visits=row["visits"])
    if seen != proofs.keys() or dict(proof_tally) != detail["summary"]["lean"]:
        raise ValueError("incomplete or inconsistent checked Lean scores")

    if graphs.keys() != detail["revision"].keys() or graphs.keys() != detail["summary"]["revision"].keys():
        raise ValueError("unexpected graph revision splits")
    for split, dataset in graphs.items():
        seen = set()
        counters = {name: defaultdict(Counter) for name in protocol["rule_policies"]}
        for row in detail["revision"][split]:
            task, targets = dataset[row["id"]]
            patch = task["patch"]
            links = dict(task["links"])
            no_op = links[patch["source"]] == patch["destination"]
            if row["id"] in seen or row["lineage"] != task["lineage"] or row["no_op"] != no_op:
                raise ValueError("duplicate or mismatched graph revision")
            seen.add(row["id"])
            if row["policies"].keys() != counters.keys():
                raise ValueError("graph revision did not exercise every registered routing policy")
            steps = task["steps"]
            original = task["start"]
            remaining = 0
            for visit in range(steps):
                if not no_op and not remaining and original == patch["source"]:
                    remaining = steps - visit
                original = links[original]
            for name, policy in row["policies"].items():
                metric = counters[name]["no_op" if no_op else "edited"]
                judged = {}
                for epoch, turn in enumerate(("before", "after")):
                    proof = policy[turn]
                    judged[turn] = proof["prediction"] == targets[turn]
                    if name != "terminal":
                        attempts = proof["attempts"]
                        reuse = name == "verified_reuse" and epoch == 1 and no_op and judged["before"]
                        visits = [0] if reuse else list(range(1, proof["visits"] + 1))
                        if (len(attempts) != proof["checks"] or not attempts or proof["visits"] > depth
                                or [a[0] for a in attempts] != visits
                                or proof["checker_hops"] != (remaining if epoch else steps)):
                            raise ValueError("graph verifier work differs from its trace")
                        if any(e != epoch or committed != (candidate == targets[turn])
                               for _, candidate, e, committed in attempts):
                            raise ValueError("graph trace contains an unsupported commit")
                        final = attempts[-1][1] if attempts[-1][-1] else None
                        if (final != proof["prediction"] or any(a[-1] for a in attempts[:-1])
                                or (final is None and proof["visits"] != depth)
                                or (reuse and (proof["visits"] != 0 or final != policy["before"]["prediction"]))):
                            raise ValueError("graph verifier failed to stop on its first certificate")
                    elif proof["checks"] or proof["visits"] != depth or proof["checker_hops"] or proof["attempts"]:
                        raise ValueError("terminal comparison used the wrong computation budget")
                    metric[turn + "_correct"] += judged[turn]
                    for field in ("visits", "checks", "checker_hops"):
                        metric[field] += proof[field]
                if judged != policy["accuracy"]:
                    raise ValueError("graph correctness differs from independent target")
                metric.update(goals=1, joint_correct=int(all(judged.values())))
        computed = {policy: {group: dict(counts) for group, counts in groups.items()}
                    for policy, groups in counters.items()}
        if seen != dataset.keys() or computed != detail["summary"]["revision"][split]:
            raise ValueError("incomplete or inconsistent graph revision evidence")


def package(lean_run, checked_pack, mixed, recipe, archive):
    lean_resource = resources(lean_run, ("evaluate", "certify"))
    mixed_resource = resources(mixed, ("prepare", "fit", "evaluate"))
    lean = load_structured(lean_run / "evidence.json")
    checked = load_structured(checked_pack / "manifest.json")
    dataset = load_structured(mixed / "dataset.json")
    evidence = load_structured(mixed / "evidence.json")
    protocol = load_structured(mixed / "fit-protocol.json")
    eval_protocol = load_structured(mixed / "eval-protocol.json")
    fits = load_structured(mixed / "fit-receipts.json")
    baselines = load_structured(mixed / "baselines.json")
    if (
        checked["source_evidence_sha256"] != sha256_file(lean_run / "evidence.json")
        or checked["resource_history_sha256"] != sha256_file(lean_run / "resource-history.jsonl")
        or dataset["sources"]["original_lean_manifest_sha256"] != sha256_file(checked_pack / "manifest.json")
        or evidence["dataset_sha256"] != sha256_file(mixed / "dataset.json")
        or evidence["fit_protocol_sha256"] != sha256_file(mixed / "fit-protocol.json")
        or evidence["eval_protocol_sha256"] != sha256_file(mixed / "eval-protocol.json")
        or eval_protocol["fit_receipts_sha256"] != sha256_file(mixed / "fit-receipts.json")
        or fits["protocol_sha256"] != sha256_file(mixed / "fit-protocol.json")
        or evidence["baselines"] != baselines
    ):
        raise ValueError("the assembled evidence has mismatched lineage")
    if not load_structured(mixed / "resume-audit.json")["identical_weight_tensors"]:
        raise ValueError("checkpoint continuation failed")
    seeds = protocol["spec"]["seeds"]
    arms = protocol["arms"]
    recipes = set(protocol["recipes"]) - {"mixed"}
    planned = protocol["spec"]["depth"]
    expected = {("mixed", s, a, a, k) for s in seeds for a in arms for k in eval_protocol["depths"]}
    expected.update(("mixed", s, a, b, planned) for s in seeds for a in arms for b in arms)
    expected.update((r, s, "direct", "direct", planned) for s in seeds for r in recipes)
    text = {r["id"]: r for r in data.json_lines(mixed / "text-held.jsonl")}
    proofs = {
        row["id"]: (row, receipt) for row, receipt in zip(
            data.json_lines(mixed / "lean-held_structures.public.jsonl"),
            data.json_lines(mixed / "lean-held_structures.receipts.jsonl"), strict=True,
        ) if receipt["native"] is not None and receipt["native"]["accepted"] is None
    }
    graphs = {
        split: {task["id"]: (task, targets) for task, targets in data.rule_rows(mixed / "revision", split)}
        for split in eval_protocol["rule_splits"]
    }
    seen = set()
    entries = {}
    for row in evidence["matrix"]:
        key = row["recipe"], row["seed"], row["fit_arm"], row["runtime_arm"], row["depth"]
        if key in seen or key not in expected:
            raise ValueError("unexpected or duplicate arm")
        seen.add(key)
        path = data.checked_path(mixed / "eval", row["trace_file"], row["trace_sha256"])
        detail = load_structured(path)
        if not math.isfinite(detail["wall_s"]) or detail["wall_s"] < 0 or detail["wall_s"] != row["wall_s"]:
            raise ValueError("mixed evaluation wall time differs from the trace")
        if detail["summary"] != row["summary"] or key != tuple(
            detail[n] for n in ("recipe", "seed", "fit_arm", "runtime_arm", "depth")
        ):
            raise ValueError("score and underlying trace disagree")
        audit_trace(detail, text, proofs, graphs, eval_protocol)
        entries[f"mixed/eval/{path.name}"] = path
    if seen != expected:
        raise ValueError("matched schedule evaluation is incomplete")
    for seed in seeds:
        conditions = [r for r in fits["arms"] if r["seed"] == seed and r["recipe"] == "mixed"]
        identity = {(r["initialization_sha256"], r["batch_plan_sha256"], r["steps"], r["parameters"])
                    for r in conditions}
        if len(conditions) != len(arms) or {r["arm"] for r in conditions} != set(arms) or len(identity) != 1:
            raise ValueError("matched training receipts are incomplete")
        for arm in conditions:
            folder = f"fits/{seed}/{arm['arm']}"
            entries[f"mixed/{folder}/weights.npz"] = data.checked_path(
                mixed / folder, "weights.npz", arm["weights_sha256"],
            )
        secondary = [r for r in fits["arms"] if r["seed"] == seed and r["recipe"] != "mixed"]
        if len(secondary) != len(recipes) or {r["recipe"] for r in secondary} != recipes:
            raise ValueError("data-mix and curriculum fits are incomplete")
        direct = next(r for r in conditions if r["arm"] == "direct")
        for control in secondary:
            if (
                control["arm"] != "direct" or control["parameters"] != direct["parameters"]
                or control["initialization_sha256"] != direct["initialization_sha256"]
                or control["steps"] != direct["steps"]
                or control["cell_calls_per_training_step"] != direct["cell_calls_per_training_step"]
                or (control["recipe"] == "text_first" and control["batch_plan_sha256"] != direct["batch_plan_sha256"])
            ):
                raise ValueError("data recipe changed architecture or planned cell calls")
            folder = f"recipe-fits/{seed}/{control['recipe']}"
            entries[f"mixed/{folder}/weights.npz"] = data.checked_path(
                mixed / folder, "weights.npz", control["weights_sha256"],
            )
    for name, digest in dataset["files"].items():
        path = mixed / name
        if not path.resolve().is_relative_to(mixed.resolve()) or sha256_file(path) != digest:
            raise ValueError("a mixed-pilot dataset member changed")
        entries[f"mixed/{name}"] = path
    rule_manifest = load_structured(mixed / "revision" / "manifest.json")
    for split, spec in rule_manifest["splits"].items():
        for kind in ("tasks", "targets"):
            name = f"{split}.{kind}.jsonl"
            entries[f"mixed/revision/{name}"] = data.checked_path(
                mixed / "revision", name, spec[f"{kind}_sha256"],
            )
    for split in checked["splits"].values():
        for kind in ("public", "receipts"):
            name = split[kind]
            entries[f"lean/data/{name}"] = data.checked_path(checked_pack, name, split[f"{kind}_sha256"])
    entries["lean/data/manifest.json"] = checked_pack / "manifest.json"
    for identifier, digest in lean["traces_sha256"].items():
        name = f"{identifier}.json"
        entries[f"lean/traces/{name}"] = data.checked_path(lean_run / "traces", name, digest)
    for batch in lean["standalone_replay"]:
        for name, key in ((batch["file"], "source_sha256"),
                          (Path(batch["file"]).with_suffix(".log").name, "output_sha256")):
            entries[f"lean/certificates/{name}"] = data.checked_path(lean_run / "certificates", name, batch[key])
    entries.update(clip_tree(lean_run, "lean", (
        "fixture.json", "library.json", "evidence.json", "protocol.json",
        "resource-history.jsonl", "witness-preflight.json",
    )))
    entries.update(clip_tree(mixed, "mixed", (
        "dataset.json", "fit-protocol.json", "fit-receipts.json", "resume-audit.json",
        "eval-protocol.json", "evidence.json", "baselines.json", "resource-history.jsonl",
    )))
    staged = mixed / "independent-lanes-audit.json"
    if staged.exists():
        stage = load_structured(staged)
        if (
            stage["job_peak_commit_bytes"] > stage["cap_bytes"] or stage["cap_bytes"] > HARD_CAP
            or stage["text_train_sha256"] != dataset["files"]["text-train.jsonl"]
            or stage["text_held_sha256"] != dataset["files"]["text-held.jsonl"]
            or stage["revision_manifest_sha256"] != dataset["files"]["revision/manifest.json"]
        ):
            raise ValueError("staged data differs from the frozen mixed dataset")
        entries["mixed/independent-lanes-audit.json"] = staged
    lean_protocol = load_structured(lean_run / "protocol.json")
    sources = {
        "rlt_lean_composition.py": lean_protocol["generator_sha256"],
        "rlt_lean_pool_audit.py": lean_protocol["source_sha256"],
        "rlt_lean_ldt.py": lean_protocol["gate_sha256"],
        "rlt_pretrain_cell.py": protocol["cell_source_sha256"],
        "rlt_pretrain_data.py": protocol["data_source_sha256"],
        "rlt_pretrain_fit.py": protocol["fit_source_sha256"],
        "rlt_pretrain_eval.py": eval_protocol["source_sha256"],
        "rlt_pilot.py": protocol["feedback_source_sha256"],
        "rlt_revision.py": dataset["sources"]["revision_generator_sha256"],
    }
    for name, digest in sources.items():
        entries[f"source/{name}"] = data.checked_path(Path(__file__).parent, name, digest)
    layout = mixed / "planning-layout.json"
    layout.write_text(pretty_json(layout_report(Layout(), planned, planned)), encoding="utf-8")
    entries["planning-layout.json"] = layout
    entries["source/rlt_pretrain_scale.py"] = Path(__file__).with_name("rlt_pretrain_scale.py")
    entries["source/rlt_pretrain_export.py"] = Path(__file__)
    entries["1B-pretraining-recipe.md"] = recipe
    audit = mixed / "bundle-audit.json"
    audit.write_text(pretty_json({
        "matched_evaluations": len(seen), "lean_resources": lean_resource, "mixed_resources": mixed_resource,
        "recipe_sha256": sha256_file(recipe),
    }), encoding="utf-8")
    entries["bundle-audit.json"] = audit
    return zip_package(archive, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("lean_run", "checked_pack", "mixed", "recipe", "archive"):
        parser.add_argument(path, type=Path)
    args = parser.parse_args()
    print(json.dumps({"files": len(package(**vars(args))["files"])}))


if __name__ == "__main__":
    main()
