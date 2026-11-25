"""Check train-only metadata and compare elaborated goal types against the public retrospective."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("verified", "retrospective", "lean", "out"):
        cli.add_argument("--" + arg, type=Path, required=True)
    args = cli.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Use a fresh audit directory")
    out.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    manifest = json.loads((args.verified / "admissions.json").read_bytes())
    records = (args.verified / "records.jsonl").read_bytes()
    if sha(records) != manifest["members"]["records.jsonl"]:
        raise ValueError("training record integrity")
    families = ast.literal_eval(
        next(
            n.value
            for n in ast.parse((here / "reserve.py").read_text(encoding="utf8")).body
            if isinstance(n, ast.Assign) and n.targets[0].id == "FAMILIES"
        )
    )
    fresh = {s for group in families.values() for s in group}
    train = set()
    degenerate = Counter()
    labels = defaultdict(Counter)
    lean_rows = []
    for line in records.splitlines():
        row = json.loads(line)
        if row["split"] != "train":
            raise ValueError("non-training row in admissions")
        train.add(row["family"].split("/", 1)[1])
        public = json.loads(row["prompt"].split("\n", 1)[1])
        target = row["target"]
        skill = row["skill"]
        if skill == "lean":
            lean_rows.append(public["statement"])
            continue
        answer = json.loads(target)
        if skill == "state":
            degenerate["state_initial_unchanged_full_trace"] += all(x == public["start"] for x in answer)
        if skill == "operator":
            x, y = public["query"]
            nearest = min(public["observations"], key=lambda v: (abs(v[0] - x) + abs(v[1] - y), v[0], v[1]))
            degenerate["operator_nearest_observation_query_value_correct"] += nearest[2] == answer["value"]
        if skill == "repair":
            labels["repair"]["equivalent" if answer["witness"] is None else "fault"] += 1
        if skill == "allocation":
            labels["allocation"][answer["solver"]] += 1
    if train & fresh or train & set(manifest["excluded_families"]["external_tournament"]):
        raise ValueError("family collision with public registered suite")
    prepared = json.loads((args.retrospective / "PREPARED.json").read_bytes())
    historical = []
    for stage in (10, 11, 12, 13, 14):
        code = (args.retrospective / f"STATE{stage}_REPLAY.lean").read_text(encoding="utf8")
        if sha(code.encode()) != prepared["prepared"][str(stage)]["source_sha256"]:
            raise ValueError("public retrospective source hash mismatch")
        declarations = re.findall(r"^theorem\s+\w+\s*(.*?)\s*:=", code, re.M | re.S)
        historical.extend((f"historical_{stage}_{i}", s) for i, s in enumerate(declarations))
    goals = [
        *historical,
        *[(f"training_{i}", ": " + stmt) for i, stmt in enumerate(lean_rows)],
        ("check_a", ": ∀ (x : Nat), x = x"),
        ("check_b", ": ∀ (y : Nat), y = y"),
        ("check_c", ": ∀ (y : Nat), y ≤ y"),
    ]
    local = out / "Types.lean"
    tool = (here / "AdmissionType.lean").read_text(encoding="utf8")
    local.write_bytes(
        (tool + "\n" + "\n".join("#signature " + name + " " + term for name, term in goals) + "\n").encode()
    )
    run = subprocess.run([str(args.lean.resolve()), "-j", "1", "Types.lean"], cwd=out, capture_output=True, timeout=360)
    (out / "elaboration.log").write_bytes(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError("type elaboration failed; see elaboration.log")
    observations = {}
    for line in run.stdout.decode("utf8").splitlines():
        if line.startswith("TYPED "):
            _, name, typ = line.split(" ", 2)
            observations[name] = sha(typ.encode())
    if (
        set(observations) != {n for n, _ in goals}
        or observations["check_a"] != observations["check_b"]
        or observations["check_a"] == observations["check_c"]
    ):
        raise ValueError("elaborated goal-type identity control failed")
    old = {observations[n] for n, _ in historical}
    new = {v for k, v in observations.items() if k.startswith("training_")}
    result = dict(
        schema="datagoblin.aleph.candidate01.public-custody-audit.v1",
        manifest_sha256=manifest["members"]["manifest.json"],
        train_families=sorted(train),
        sealed_math_families=sorted(fresh),
        held_tournament_families=manifest["excluded_families"]["external_tournament"],
        labels={k: dict(v) for k, v in labels.items()},
        public_train_cues=dict(degenerate),
        historical_theorem_declarations=len(historical),
        unique_historical_elaborated_types=len(old),
        admitted_lean=len(lean_rows),
        unique_admitted_types=len(new),
        shared_closed_type_count=len(old & new),
        type_checker_sha256=sha(tool.encode()),
        prepared_public_sha256=sha((args.retrospective / "PREPARED.json").read_bytes()),
        private_evaluation_files_read=0,
    )
    (out / "audit.json").write_bytes((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(result, sort_keys=True))
    if old & new or len(new) != len(lean_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
