"""Schedules and independent Sudoku/LDT scoring for the released HRM pilot."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

ARMS = ("direct", "segment_rlt", "global_rlt", "global_damping")
DIGITS = frozenset("123456789")
UNITS = (
    tuple(tuple(range(i, i + 9)) for i in range(0, 81, 9))
    + tuple(tuple(range(i, 81, 9)) for i in range(9))
    + tuple(tuple(9 * (r + a) + c + b for a in range(3) for b in range(3))
            for r in (0, 3, 6) for c in (0, 3, 6))
)


def slow_coefficients(arm: str, segments: int, h_cycles: int = 2) -> tuple[float, ...]:
    total = segments * h_cycles
    if arm not in ARMS or total < 1 or segments < 1 or h_cycles < 1:
        raise ValueError("invalid slow feedback schedule")
    if arm == "direct" or total == 1:
        return (1.0,) * total
    if arm == "segment_rlt":
        return tuple(1 / (h_cycles - j) for j in range(h_cycles)) * segments
    weights = tuple(1 / j for j in range(total, 0, -1))
    if arm == "global_rlt":
        return weights
    return (math.fsum(weights[:-1]) / (total - 1),) * (total - 1) + (1.0,)


def sudoku_valid(givens: str, proposed: str) -> bool:
    if len(givens) != 81 or len(proposed) != 81 or set(proposed) - DIGITS:
        return False
    if any(g != "0" and g != p for g, p in zip(givens, proposed, strict=True)):
        return False
    return all({proposed[i] for i in unit} == DIGITS for unit in UNITS)


def report(root: Path, out: Path, upstream_ldt: Path):
    resources = load_structured(root / "resources.json")
    if (root / "active.lock").exists() or resources["segments"][-1]["status"] != "complete":
        raise ValueError("inference must finish before reporting")
    if any(r["peak_commit_bytes"] > r["hard_commit_cap_bytes"] for r in resources["segments"]):
        raise ValueError("observed job use exceeds the cap")
    protocol = load_structured(root / "protocol.json")
    holdout_path = root / "holdout.json"
    if sha256_file(holdout_path) != protocol["holdout_sha256"]:
        raise ValueError("the frozen holdout changed")
    holdout = load_structured(holdout_path)
    sys.path.insert(0, str(upstream_ldt.resolve()))
    from research_gym.core import hybrid

    ldt_hash = sha256_file(Path(hybrid.__file__))
    segments = protocol["segments"]
    cases = []
    for item in holdout["puzzles"]:
        givens, solution = item["givens"], item["solution"]
        if not sudoku_valid(givens, solution):
            raise ValueError("invalid verified holdout target")
        top = hybrid.CandidateState({
            str(i): DIGITS if digit == "0" else frozenset({digit}) for i, digit in enumerate(givens)
        })
        for arm in ARMS:
            prediction_path = root / "predictions" / f"{item['id']}-{arm}.json"
            record = load_structured(prediction_path)
            if record["id"] != item["id"] or record["arm"] != arm or len(record["steps"]) != segments:
                raise ValueError("wrong prediction item or segment coverage")
            first_verified = None
            verified_exact = False
            valid, exact = [], []
            for step, row in enumerate(record["steps"], 1):
                if step != row["segment"]:
                    raise ValueError("incorrect segment order")
                grid = row["prediction"]
                valid.append(sudoku_valid(givens, grid))
                exact.append(grid == solution)
                if first_verified is not None:
                    continue
                proposal = hybrid.LatticeProposal(
                    hybrid.CandidateState({str(i): frozenset({digit}) for i, digit in enumerate(grid)}),
                    hybrid.SoundnessType.MODEL_SOUND_DEAD,
                )
                evidence = hybrid.SoundnessType.ENV_SOUND_DEAD if valid[-1] else hybrid.SoundnessType.UNKNOWN
                check = hybrid.certify_and_apply_verified(
                    top, proposal, provenance_verifier=lambda _p, _c, verdict=evidence: verdict,
                )
                if check.accepted != valid[-1] or check.after != (proposal.proposed_state if valid[-1] else top):
                    raise ValueError("LDT boundary disagrees with independent Sudoku validation")
                if check.accepted:
                    first_verified = step
                    verified_exact = exact[-1]
            final = record["steps"][-1]["prediction"]
            cases.append({
                "id": item["id"], "arm": arm, "first_verified_segment": first_verified,
                "gate_exact": verified_exact, "gate_checks": first_verified or segments,
                "valid_by_segment": valid, "exact_by_segment": exact,
                "final_cell_errors": sum(x != y for x, y in zip(final, solution, strict=True)),
                "predictions_sha256": sha256_file(prediction_path),
            })
    summaries = {}
    calls = protocol["h_cycles"] * (1 + protocol["l_cycles"])
    for arm in ARMS:
        rows = [r for r in cases if r["arm"] == arm]
        summaries[arm] = {
            "items": len(rows), "final_cell_errors": sum(r["final_cell_errors"] for r in rows),
            "valid_by_segment": [sum(r["valid_by_segment"][i] for r in rows) for i in range(segments)],
            "exact_by_segment": [sum(r["exact_by_segment"][i] for r in rows) for i in range(segments)],
            "verified_gate_exact": sum(r["gate_exact"] for r in rows),
            "verified_gate_valid": sum(r["first_verified_segment"] is not None for r in rows),
            "gate_board_checks": sum(r["gate_checks"] for r in rows),
            "full_stack_calls": len(rows) * segments * calls,
            "gate_replay_stack_calls": sum(r["gate_checks"] for r in rows) * calls,
        }
    dump_structured(out, {
        "schema": "datagoblin.rlt-hrm-sudoku-checkpoint.v1", "protocol": protocol, "resources": resources,
        "upstream_ldt_sha256": ldt_hash, "puzzle_count": len(holdout["puzzles"]), "summary": summaries, "cases": cases,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream-ldt", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.out, args.upstream_ldt)


if __name__ == "__main__":
    main()
