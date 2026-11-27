"""ZKL-11: topology-separated deletion tasks, with a pinned external identity checker."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from datagoblin.zk_lean_gym import lc, public_relation, shape_digest

BN254_FR = 21888242871839275222246405745257275088548364400416034343698204186575808495617
DEVELOPMENT = ("linear_chain", "scaled_product", "essential_linear")
HOLDOUT = ("quadratic_sum", "multiplicative_implication", "essential_quadratic")


def make_case(p, family, rng):
    def linear(*terms):
        return [lc(p, *terms), lc(p, (0, 1)), []]

    def product(a, b, c):
        return [lc(p, *a), lc(p, *b), lc(p, *c)]

    a, b = (rng.randrange(1, p) for _ in range(2))
    span, equivalent = True, True
    if family == "linear_chain":
        rows = [linear((2, 1), (1, -a)), linear((3, 1), (2, -b)),
                linear((3, 1), (1, -a * b))]
    elif family == "scaled_product":
        rows = [product([(1, 1)], [(2, 1)], [(3, 1)]),
                product([(2, a)], [(1, b)], [(3, a * b)])]
    elif family == "essential_linear":
        rows = [linear((1, 1), (2, -1)), linear((2, 1), (3, -1))]
        span = equivalent = False
    elif family == "quadratic_sum":
        rows = [product([(1, 1)], [(2, 1)], [(3, 1)]),
                product([(1, 1)], [(3, 1)], [(2, 1)]),
                product([(1, 1)], [(2, a), (3, b)], [(3, a), (2, b)])]
    elif family == "multiplicative_implication":
        rows = [linear((1, 1)), product([(1, 1)], [(2, a)], [])]
        span = False  # a nonconstant multiplier is needed
    elif family == "essential_quadratic":
        rows = [product([(1, 1)], [(2, 1)], [(3, 1)]),
                product([(1, 1)], [(3, 1)], [(2, 1)])]
        span = equivalent = False
    else:
        raise ValueError(family)
    s = {"format": "zkl-r1cs-v1", "modulus": str(p), "wires": 4,
         "public": [1, 2, 3], "rows": rows}
    replacement = dict(s, rows=rows[:-1])
    return {"original": s, "replacement": replacement, "target_index": len(rows) - 1,
            "construction_expected_span": span, "construction_expected_equivalent": equivalent}


def cases(seed, split, per_family, p):
    if split not in ("development", "holdout") or per_family < 1 or p not in (7, 11, BN254_FR):
        raise ValueError("invalid campaign parameters")
    rng = random.Random(seed)
    families = DEVELOPMENT if split == "development" else HOLDOUT
    for family in families:
        for root in range(per_family):
            case = make_case(p, family, rng)
            yield {"schema": "zkl-ideal-gym-v1", "seed": seed, "split": split,
                   "family": family, "root": root, **case}


def finite_receipt(case):
    """Independent complete assignment relation check; all nonconstant wires are public."""
    a, b = case["original"], case["replacement"]
    p = int(a["modulus"])
    if p not in (7, 11):
        return {"status": "NOT_ENUMERATED", "equivalent": None, "counterexample": None}
    before, after = public_relation(a), public_relation(b)
    extra = sorted(after.keys() - before.keys())
    missing = sorted(before.keys() - after.keys())
    if missing:
        raise AssertionError("deletion lost accepted assignment")
    return {"status": "EXHAUSTIVE_FINITE_DOMAIN", "equivalent": not extra,
            "counterexample": None if not extra else {"public": extra[0], "assignment": after[extra[0]]}}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_external(case, checker, checker_sha256, audit_sha256, timeout=15):
    """Only invokes an operator-pinned checker checkout; no shell or model calls."""
    checker = checker.resolve()
    if (checker.name != "ideal_certificate.py" or file_sha(checker) != checker_sha256
            or file_sha(checker.with_name("audit.py")) != audit_sha256):
        raise ValueError("external checker pin mismatch")
    a, b = case["original"], case["replacement"]
    ha, hb = shape_digest(a), shape_digest(b)
    with tempfile.TemporaryDirectory(prefix="zkl-ideal-") as work:
        root = Path(work)
        src, dst = root / "original.json", root / "replacement.json"
        proof, receipt_file = root / "Ideal.lean", root / "receipt.json"
        src.write_text(json.dumps(a), encoding="utf-8")
        dst.write_text(json.dumps(b), encoding="utf-8")
        run = subprocess.run(
            [sys.executable, "-B", str(checker), str(src), str(dst),
             "--target", str(case["target_index"]), "--expected-original", ha,
             "--expected-replacement", hb, "--lean-out", str(proof), "--receipt-out", str(receipt_file)],
            cwd=root, capture_output=True, text=True, timeout=timeout, check=False,
        )
        if run.returncode:
            if (run.returncode == 1
                    and run.stderr.strip() == "INCONCLUSIVE: target is outside the retained quadratic span"
                    and not proof.exists() and not receipt_file.exists()):
                return {"status": "INCONCLUSIVE", "exact_identity_checked": False,
                        "lean_replayed": False, "production_accepted": False}
            raise RuntimeError(f"checker failed: {run.stderr[:1000]}")
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        cert = receipt["certificate"]
        if (receipt["exact_identity_checked"] is not True or receipt["lean_replayed"] is not False
                or receipt["production_accepted"] is not False or cert["original_sha256"] != ha
                or cert["replacement_sha256"] != hb or cert["target_index"] != case["target_index"]
                or receipt["lean_source_sha256"] != file_sha(proof)):
            raise ValueError("invalid checker receipt")
        return {"status": "EXACT_IDENTITY_CHECKED", **receipt,
                "checker_sha256": checker_sha256, "audit_sha256": audit_sha256,
                "lean_source": proof.read_text(encoding="utf-8")}


def campaign(seed, split, per_family, p, checker=None, checker_sha256=None, audit_sha256=None):
    if checker is not None and (not checker_sha256 or not audit_sha256):
        raise ValueError("both external source pins required")
    for case in cases(seed, split, per_family, p):
        finite = finite_receipt(case)
        if finite["equivalent"] is not None and finite["equivalent"] != case["construction_expected_equivalent"]:
            raise AssertionError("fixture construction disagrees with exhaustive relation")
        receipt = ({"status": "NOT_RUN"} if checker is None else
                   checked_external(case, checker, checker_sha256, audit_sha256))
        if receipt["status"] != "NOT_RUN":
            if (receipt["status"] == "EXACT_IDENTITY_CHECKED") != case["construction_expected_span"]:
                raise AssertionError("fixture expected span disagrees with checker")
            if receipt["status"] == "EXACT_IDENTITY_CHECKED" and finite["equivalent"] is False:
                raise AssertionError("checker accepted an unsound deletion")
        yield {**case, "finite_receipt": finite, "identity_receipt": receipt,
               "production_accepted": False, "lean_replayed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--field", choices=("7", "11", "bn254"), required=True)
    parser.add_argument("--checker", type=Path)
    parser.add_argument("--checker-sha256")
    parser.add_argument("--audit-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p = BN254_FR if args.field == "bn254" else int(args.field)
    counts = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        for rec in campaign(args.seed, args.split, args.per_family, p, args.checker,
                            args.checker_sha256, args.audit_sha256):
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            status = rec["identity_receipt"]["status"]
            counts[status] = counts.get(status, 0) + 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
