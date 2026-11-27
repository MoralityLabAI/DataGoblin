"""ZKL-12: bounded polynomial certificate curriculum and independent finite controls."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from datagoblin.zk_ideal_gym import BN254_FR, file_sha, finite_receipt, make_case
from datagoblin.zk_lean_gym import lc, shape_digest

DEVELOPMENT = ("linear_chain", "multiplicative_implication", "essential_linear")
HOLDOUT = ("scaled_product", "inverse_transport", "square_transport", "essential_quadratic")
EXPECTED_DEGREE = {
    "linear_chain": 0, "multiplicative_implication": 1, "essential_linear": None,
    "scaled_product": 0, "inverse_transport": 1, "square_transport": 2,
    "essential_quadratic": None,
}


def build_case(p, family, rng):
    if family not in ("inverse_transport", "square_transport"):
        base = make_case(p, family, rng)
    else:
        x, y, z = (lc(p, (i, 1)) for i in (1, 2, 3))
        one = lc(p, (0, 1))
        k = rng.randrange(1, p)
        if family == "inverse_transport":
            rows = [[x, y, one], [y, z, one], [lc(p, (1, k), (3, -k)), one, []]]
        else:
            rows = [[x, y, z], [x, x, []], [z, lc(p, (3, k)), []]]
        original = {"format": "zkl-r1cs-v1", "modulus": str(p), "wires": 4,
                    "public": [1, 2, 3], "rows": rows}
        base = {"original": original, "replacement": dict(original, rows=rows[:-1]),
                "target_index": len(rows) - 1, "construction_expected_equivalent": True}
    return {**base, "construction_expected_degree": EXPECTED_DEGREE[family]}


def cases(seed, split, per_family, p):
    if split not in ("development", "holdout") or per_family < 1 or p not in (7, 11, BN254_FR):
        raise ValueError("invalid campaign parameters")
    rng = random.Random(seed)
    families = DEVELOPMENT if split == "development" else HOLDOUT
    for family in families:
        for root in range(per_family):
            yield {"schema": "zkl-poly-gym-v1", "seed": seed, "split": split,
                   "family": family, "root": root, **build_case(p, family, rng)}


def checked_external(case, checker, pins, degree, timeout=15):
    """Invoke only reviewed, pinned files. Proof replay is still a separate gate."""
    checker = checker.resolve()
    names = ("poly_certificate.py", "ideal_certificate.py", "audit.py")
    if checker.name != names[0] or set(pins) != set(names):
        raise ValueError("three checker source pins required")
    if any(file_sha(checker.with_name(name)) != pins[name] for name in names):
        raise ValueError("external checker pin mismatch")
    a, b = case["original"], case["replacement"]
    ha, hb = shape_digest(a), shape_digest(b)
    with tempfile.TemporaryDirectory(prefix="zkl-poly-") as work:
        root = Path(work)
        src, dst = root / "original.json", root / "replacement.json"
        proof, receipt_file = root / "Poly.lean", root / "receipt.json"
        src.write_text(json.dumps(a), encoding="utf-8")
        dst.write_text(json.dumps(b), encoding="utf-8")
        run = subprocess.run(
            [sys.executable, "-B", "-E", "-s", str(checker), str(src), str(dst),
             "--target", str(case["target_index"]), "--degree", str(degree),
             "--expected-original", ha, "--expected-replacement", hb,
             "--lean-out", str(proof), "--receipt-out", str(receipt_file)],
            cwd=root, capture_output=True, text=True, timeout=timeout, check=False,
        )
        if run.returncode:
            if (run.returncode == 1
                    and run.stderr.strip() == "INCONCLUSIVE: target is outside the bounded monomial span"
                    and not proof.exists() and not receipt_file.exists()):
                return {"status": "INCONCLUSIVE", "degree_budget": degree, "lean_replayed": False,
                        "production_accepted": False}
            raise RuntimeError(f"checker failed: {run.stderr[:1000]}")
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        cert = receipt["certificate"]
        if (receipt["exact_identity_checked"] is not True or receipt["lean_replayed"] is not False
                or receipt["production_accepted"] is not False or cert["original_sha256"] != ha
                or cert["replacement_sha256"] != hb or cert["target_index"] != case["target_index"]
                or type(cert["degree"]) is not int or not 0 <= cert["degree"] <= degree
                or receipt["lean_source_sha256"] != file_sha(proof)):
            raise ValueError("invalid checker receipt")
        return {"status": "EXACT_IDENTITY_CHECKED", "degree_budget": degree, **receipt,
                "source_pins": pins, "lean_source": proof.read_text(encoding="utf-8")}


def campaign(seed, split, per_family, p, checker=None, pins=None):
    if checker is not None and pins is None:
        raise ValueError("checker source pins required")
    for case in cases(seed, split, per_family, p):
        finite = finite_receipt(case)
        if finite["equivalent"] is not None and finite["equivalent"] != case["construction_expected_equivalent"]:
            raise AssertionError("fixture construction disagrees with exhaustive relation")
        receipts = []
        if checker is not None:
            for degree in range(3):
                receipt = checked_external(case, checker, pins, degree)
                expected = case["construction_expected_degree"]
                found = receipt["status"] == "EXACT_IDENTITY_CHECKED"
                if found != (expected is not None and expected <= degree):
                    raise AssertionError("degree tier disagrees with fixture")
                if found and (receipt["certificate"]["degree"] != expected or finite["equivalent"] is False):
                    raise AssertionError("wrong minimum degree or unsound deletion")
                receipts.append(receipt)
        yield {**case, "finite_receipt": finite, "tier_receipts": receipts,
               "production_accepted": False, "lean_replayed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--field", choices=("7", "11", "bn254"), required=True)
    parser.add_argument("--checker", type=Path)
    parser.add_argument("--poly-sha256")
    parser.add_argument("--ideal-sha256")
    parser.add_argument("--audit-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p = BN254_FR if args.field == "bn254" else int(args.field)
    pins = {"poly_certificate.py": args.poly_sha256, "ideal_certificate.py": args.ideal_sha256,
            "audit.py": args.audit_sha256}
    counts = {"cases": 0, "certified_d0": 0, "certified_d1": 0, "certified_d2": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        for rec in campaign(args.seed, args.split, args.per_family, p, args.checker, pins):
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            counts["cases"] += 1
            for receipt in rec["tier_receipts"]:
                if receipt["status"] == "EXACT_IDENTITY_CHECKED":
                    counts[f"certified_d{receipt['degree_budget']}"] += 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
