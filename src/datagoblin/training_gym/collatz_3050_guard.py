"""Protect exposed BPD-COL-02 roots from subsequent curriculum training."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Any

from datagoblin.training_gym.collatz_corpus import make_certificate
from datagoblin.training_gym.collatz_verifier import verify_certificate

SOURCE = "f1d489a000406e2d8732f4ea7a39e5473392efe2"
ROLES = ("recipient", "effect_donor", "sham_donor")


def digest(m: dict[str, Any]) -> str:
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(raw.encode()).hexdigest()


def branch(n: int) -> int:
    return 3 if n % 4 == 1 else 1


def step(d: int, n: int) -> int:
    n = branch(n) * n + d
    while n % 2 == 0:
        n //= 2
    return n


def quotient(d: int, n: int) -> int:
    seen: set[int] = set()
    pending = [n % d]
    while pending:
        x = pending.pop()
        if x not in seen:
            seen.add(x)
            pending.extend((2 * x % d, 3 * x % d))
    return min(seen)


def build_guard(manifest: dict[str, Any]) -> dict[str, Any]:
    """Replay the exact cases and retain parameter exclusions, never training rows."""
    if (
        manifest["schema"] != "BPD-COL-02/public-development-cases/v1"
        or manifest["manifest_sha256"] != digest(manifest)
        or manifest["specification"]["split"] != "development_only"
        or manifest["specification"]["confirmation_authorized"] is not False
        or manifest["source"]["commit"] != "b3eb33ed10485264fe7a593e3b0ace418d9f9a6d"
        or manifest["source"]["oracle_sha256"] != "24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca"
    ):
        raise ValueError("manifest source/exposure contract")
    roots: set[int] = set()
    normalized: set[int] = set()
    ids: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for c in manifest["cases"]:
        d, h = c["d"], c["horizon"]
        if type(d) is not int or not 1 <= d <= 1025 or d % 4 != 1 or gcd(d, 6) != 1:
            raise ValueError("invalid source parameter")
        if type(h) is not int or h not in (1, 2) or c["root_id"] != d:
            raise ValueError("invalid horizon/root")
        tier = c["tier"]
        cid = f"COL3050-d{d:06d}-h{h}-{tier}"
        if tier not in ("arithmetic", "trace") or cid != c["case_id"] or cid in ids:
            raise ValueError("case identity")
        ids.add(cid)
        roots.add(d)
        labels = c["label_map"]
        if set(labels) != {"1", "3"} or set(labels.values()) != {"A", "B"}:
            raise ValueError("invalid label mapping")
        ns, ys, basins = [], [], []
        for role in ROLES:
            x = c[role]
            n = x["n"]
            if type(n) is not int or not 1 <= n <= d or n % 2 != 1:
                raise ValueError("invalid odd seed")
            trace = [n]
            for _ in range(h):
                trace.append(step(d, trace[-1]))
            y = labels[str(branch(trace[-1]))]
            if trace != x["trace"] or x["label"] != y or x["future_branch"] != branch(trace[-1]):
                raise ValueError("false trajectory label")
            if c["current_observer"] != {"quotient": quotient(d, n), "branch_multiplier": branch(n)}:
                raise ValueError("observer mismatch")
            checked = verify_certificate(make_certificate(d, n, max_edges=d), max_edges=d)
            if checked["cycle_minimum"] != x["basin"]:
                raise ValueError("basin mismatch")
            reduced = d // gcd(d, n)
            normalized.add(reduced)
            ns.append(n)
            ys.append(y)
            basins.append(x["basin"])
            receipts.append({
                "case_id": cid, "role": role, "d": d, "seed": n, "normalized_d": reduced,
                "certificate_sha256": checked["certificate_sha256"],
                "checked_full_map_steps": checked["checked_full_map_steps"],
            })
        if len(set(ns)) != 3 or ys[0] != ys[2] or ys[0] == ys[1]:
            raise ValueError("donor relationship mismatch")
        if c["all_same_basin"] is not (len(set(basins)) == 1):
            raise ValueError("basin stratum mismatch")
    if roots != set(manifest["root_ids"]) or len(roots) != manifest["n_roots"] or len(ids) != manifest["n_cases"]:
        raise ValueError("manifest denominators")
    if not 1 <= len(roots) <= 32 or not 1 <= len(ids) <= 64:
        raise ValueError("manifest resource bounds")
    return {
        "schema": "DataGoblin/COL3050-quarantine/v1",
        "source": SOURCE, "manifest_sha256": manifest["manifest_sha256"],
        "exposed_raw_parameters": sorted(roots),
        "exposed_normalized_parameters": sorted(normalized | roots),
        "verified_role_count": len(receipts),
        "checked_full_map_steps": sum(x["checked_full_map_steps"] for x in receipts),
        "role_receipts": receipts,
        "training_rows_admitted": 0, "token_quota_credit": 0,
        "scope": "public development exclusion guard; no sealed confirmation or causal supervision",
    }


def check_corpus(guard: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Conservative root-family quarantine on existing verified COL corpus rows."""
    forbidden = set(guard["exposed_normalized_parameters"])
    blocked, retained = [], []
    seen: set[str] = set()
    for row in rows:
        cert = row["certificate"]
        checked = verify_certificate(cert)
        identity = checked["case_id"]
        if identity in seen or row["id"] != identity or row["split"] != checked["split"]:
            raise ValueError("duplicate or mislabelled corpus row")
        seen.add(identity)
        d, seed = cert["d"], cert["seed"]
        if row["split"] == "train":
            (blocked if d in forbidden or d // gcd(d, seed) in forbidden else retained).append(identity)
    return {
        "checked_rows": len(rows), "blocked_training_ids": sorted(blocked),
        "nonoverlapping_training_ids": sorted(retained),
        "training_overlap_count": len(blocked), "passed": not blocked,
        "scope": "root quarantine only; passing does not establish training admission",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--check-corpus", type=Path)
    args = p.parse_args()
    guard = build_guard(json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.check_corpus:
        rows = [json.loads(s) for s in args.check_corpus.read_text(encoding="utf-8").splitlines()]
        guard["corpus_audit"] = check_corpus(guard, rows)
    print(json.dumps(guard, indent=2, sort_keys=True))
    if guard.get("corpus_audit", {}).get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
