"""Reconcile every archived theorem, control, byte hash and charged Lean process."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from state15_kernel import digest, proof_axioms
from state15_replay import EXPECTED, HEADER, NAME, PINS, read_checked


def costs(records):
    return {
        "queries": len(records),
        **{
            key: {"sum": sum(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
            for key in ("wall_seconds", "cpu_seconds")
            if (v := [r[key] for r in records])
        },
    }


def compare_file(path, content, receipt):
    if path.read_bytes() != content or digest(content) != receipt["source_sha256"]:
        raise ValueError(f"query source changed: {path.name}")
    log = path.with_suffix(".log").read_bytes()
    if digest(log) != receipt["log_sha256"]:
        raise ValueError(f"query log changed: {path.name}")
    if receipt["cpu_seconds"] < 0 or receipt["wall_seconds"] <= 0:
        raise ValueError(f"unmeasured query: {path.name}")
    return log.decode("utf-8").replace("\r\n", "\n")


def audit(root, out):
    receipt_path = out / "KERNEL_RECEIPTS.json"
    raw = receipt_path.read_bytes()
    report = json.loads(raw)
    if report["archive_pins"] != {str(n): v for n, v in PINS.items()}:
        raise ValueError("archive pins changed")
    width = report["batch_size"]
    by_stage = {}
    axiom_sets = Counter()
    failed = []
    batches = {(r["stage"], r["start"]): r for r in report["positive_batches"]}
    consumed = 0
    for stage, archives in PINS.items():
        for filename, pin in archives.items():
            read_checked(root / f"aleph_state{stage}" / filename, pin)
        data = (out / f"STATE{stage}_REPLAY.lean").read_bytes()
        if digest(data) != report["prepared"][str(stage)]["source_sha256"]:
            raise ValueError("corrupt prepared theorem source")
        import re

        text = re.sub(r"^#print axioms .*$\n?", "", data.decode(), flags=re.M)
        bounds = list(NAME.finditer(text))
        if len(bounds) != EXPECTED[stage]:
            raise ValueError("theorem count mismatch")
        observed = []
        passed = 0
        for start in range(0, len(bounds), width):
            item = batches[stage, start]
            batch = bounds[start : start + width]
            ns = [m[1] for m in batch]
            ts = [
                text[m.start() : bounds[start + j + 1].start() if start + j + 1 < len(bounds) else len(text)]
                for j, m in enumerate(batch)
            ]
            source = HEADER + "\n".join(ts) + "\n" + "\n".join("#print axioms " + n for n in ns) + "\n"
            log = compare_file(out / "batches" / f"s{stage}_{start:04d}.lean", source.encode(), item)
            audited, ax = proof_axioms(log, ns)
            ok = not item["timed_out"] and item["exit_code"] == 0 and audited
            if item["passed"] != ok or ax != item["axioms"] or len(batch) != item["declarations"]:
                raise ValueError("recomputed positive outcome differs")
            if ok:
                passed += len(batch)
                axiom_sets.update(",".join(a) or "none" for a in ax.values())
            else:
                failed.append({"stage": stage, "start": start, "kind": "positive", "log_sha256": item["log_sha256"]})
            observed.append(item)
            consumed += 1
        by_stage[str(stage)] = {"declarations": len(bounds), "passed": passed, "positive_queries": costs(observed)}
    if consumed != len(report["positive_batches"]):
        raise ValueError("extra or duplicate positive receipt")
    controls = []
    for stage in (10, 11):
        for c in json.loads((root / f"aleph_state{stage}" / f"STATE{stage}_NEGATIVE.json").read_bytes()):
            controls.append((stage, c["id"], c["context"] if stage == 10 else c["source"], c["invalid"]))
    categories = {"valid_context": [], "invalid_transport": []}
    rejected = 0
    for i, ((stage, label, context, bad), item) in enumerate(zip(controls, report["negative_checks"], strict=True)):
        if (stage, label) != (item["stage"], item["id"]):
            raise ValueError("negative identity mismatch")
        src = HEADER + context
        prefix = out / "batches" / f"negative_{i:03d}"
        compare_file(prefix.with_name(prefix.name + "_context.lean"), src.encode(), item["context"])
        log = compare_file(
            prefix.with_name(prefix.name + "_invalid.lean"), (src + "\n" + bad).encode(), item["invalid"]
        )
        ok = (
            item["context"]["exit_code"] == 0
            and item["invalid"]["exit_code"] != 0
            and not (item["context"]["timed_out"] or item["invalid"]["timed_out"])
            and "type mismatch" in log.lower()
        )
        if ok != item["passed"]:
            raise ValueError("recomputed negative outcome differs")
        rejected += int(ok)
        if not ok:
            failed.append(
                {"stage": stage, "index": i, "kind": "invalid_control", "log_sha256": item["invalid"]["log_sha256"]}
            )
        for kind, k in (("valid_context", "context"), ("invalid_transport", "invalid")):
            categories[kind].append(item[k])
    positive = sum(r["passed"] for r in by_stage.values())
    if (
        positive != report["positive_declarations_passed"]
        or rejected != report["negative_rejections_passed"]
        or bool(report["kernel_verified"]) != (positive == sum(EXPECTED.values()) and rejected == 136)
    ):
        raise ValueError("terminal decision differs")
    return {
        "schema": "datagoblin.state15.kernel-audit.v1",
        "kernel_verified": report["kernel_verified"],
        "receipt_sha256": digest(raw),
        "lean_version": report["lean_version"],
        "lean_executable_sha256": report["toolchain"]["executable_sha256"],
        "batch_size": width,
        "stage_counts_and_costs": by_stage,
        "positive_declarations_passed": positive,
        "negative_controls_rejected": rejected,
        "axiom_sets": dict(axiom_sets),
        "failed_checks": failed,
        "query_costs": {
            "positive_batches": costs(report["positive_batches"]),
            **{k: costs(v) for k, v in categories.items()},
        },
        "total_process_queries": len(report["positive_batches"]) + len(controls) * 2,
        "harness_wall_seconds": report["replay_wall_seconds"],
        "training_admission": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    args = p.parse_args()
    result = audit(args.archive_root, args.out)
    args.summary.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(
        result["kernel_verified"],
        result["positive_declarations_passed"],
        result["negative_controls_rejected"],
        result["total_process_queries"],
    )
