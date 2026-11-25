"""Bounded offline arithmetic curriculum with independent replay admission.

Usage: python -m datagoblin.training_gym.collatz_corpus generate --out runs/col
       python -m datagoblin.training_gym.collatz_corpus audit --out runs/col
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import gcd
from pathlib import Path

from datagoblin.training_gym.collatz_verifier import (
    CERT_SCHEMA,
    canonical_hash,
    parameter_split,
    verify_certificate,
)


def make_certificate(d: int, seed: int, *, max_edges: int = 100_000) -> dict:
    split = parameter_split(d)
    if type(seed) is not int or seed < 1 or seed % 2 != 1:
        raise ValueError("seed must be a positive odd integer")
    if type(max_edges) is not int or max_edges < 1:
        raise ValueError("max_edges must be positive")
    states, edges, seen = [], [], {}
    n = seed
    while n not in seen:
        if len(edges) >= max_edges:
            raise ValueError("trace exceeds max_edges")
        seen[n] = len(states)
        states.append(n)
        u = int(n % 4 == 1)
        numerator = (3 if u else 1) * n + d
        a = (numerator & -numerator).bit_length() - 1
        edges.append({"halvings": a, "triple": u})
        n = numerator >> a
    states.append(n)
    offset = seen[n]
    a = sum(edge["halvings"] for edge in edges[offset:])
    u = sum(edge["triple"] for edge in edges[offset:])
    r = len(edges) - offset
    return {"schema": CERT_SCHEMA, "d": d, "seed": seed,
            "root_id": f"d:{d}", "split": split, "odd_states": states,
            "edges": edges, "cycle_start": offset,
            "claim": {"cycle_minimum": min(states[offset:-1]), "odd_steps": r,
                      "halvings": a, "triple_branches": u, "period": r + a,
                      "gcd_with_d": gcd(d, seed)}}


def admitted_record(d: int, seed: int) -> dict:
    certificate = make_certificate(d, seed)
    admission = verify_certificate(certificate)
    return {"id": admission["case_id"], "root_id": admission["root_id"],
            "split": admission["split"], "input": {"d": d, "n": seed},
            "target": {"cycle_minimum": admission["cycle_minimum"]},
            "certificate": certificate, "admission": admission,
            "supervision_scope": "finite arithmetic trajectory and cycle closure"}


def audit_records(records: list[dict]) -> dict:
    """Replay admission, root custody and normalized trace deduplication."""
    ids: set[str] = set()
    root_splits: dict[str, str] = {}
    normalized: dict[str, str] = {}
    split_counts: Counter = Counter()
    odd_edges = full_steps = 0
    for row in records:
        if type(row) is not dict:
            raise ValueError("dataset record must be an object")
        if row.get("supervision_scope") != "finite arithmetic trajectory and cycle closure":
            raise ValueError("supervision scope exceeds the certificate contract")
        admission = verify_certificate(row.get("certificate"))
        if canonical_hash(row.get("admission")) != canonical_hash(admission):
            raise ValueError("admission was not produced by current independent replay")
        identity, root, split = admission["case_id"], admission["root_id"], admission["split"]
        if row.get("id") != identity or row.get("root_id") != root or row.get("split") != split:
            raise ValueError("dataset identity does not match replay")
        cert = row["certificate"]
        if canonical_hash(row.get("input")) != canonical_hash({"d": cert["d"], "n": cert["seed"]}):
            raise ValueError("prompt inputs differ from certificate")
        target = row.get("target")
        if (type(target) is not dict or set(target) != {"cycle_minimum"}
                or type(target["cycle_minimum"]) is not int
                or target["cycle_minimum"] != admission["cycle_minimum"]):
            raise ValueError("target was not established by replay")
        if identity in ids:
            raise ValueError("duplicate task identity")
        ids.add(identity)
        if root in root_splits and root_splits[root] != split:
            raise ValueError("one parameter root appears in multiple splits")
        root_splits[root] = split
        g = cert["claim"]["gcd_with_d"]
        fingerprint = canonical_hash({"d": cert["d"] // g,
                                      "states": [n // g for n in cert["odd_states"]]})
        if fingerprint in normalized and normalized[fingerprint] != split:
            raise ValueError("scaled duplicate trajectory crosses splits")
        normalized[fingerprint] = split
        split_counts[split] += 1
        odd_edges += admission["checked_odd_edges"]
        full_steps += admission["checked_full_map_steps"]
    return {"schema": "DataGoblin/COL-AUDIT/v1", "records": len(records),
            "parameter_roots": len(root_splits), "records_per_split": dict(sorted(split_counts.items())),
            "checked_odd_edges": odd_edges, "checked_full_map_steps": full_steps,
            "canonical_dataset_sha256": canonical_hash(records),
            "duplicate_ids": 0, "cross_split_roots": 0, "cross_split_scaled_traces": 0}


def generate_records(max_d: int = 1021, per_parameter: int = 8) -> tuple[list[dict], dict]:
    if type(max_d) is not int or not 1 <= max_d <= 4093:
        raise ValueError("max_d must lie in 1..4093 for the bounded CPU fixture")
    if type(per_parameter) is not int or not 1 <= per_parameter <= 32:
        raise ValueError("per_parameter must lie in 1..32")
    records, skipped = [], 0
    for d in range(1, max_d + 1, 4):
        if gcd(d, 6) != 1:
            continue
        if parameter_split(d) == "holdout":
            skipped += 1
            continue
        # Unit seeds make the effective parameter d/gcd(n,d) equal to d.
        seeds = [n for n in range(1, d + 1, 2) if gcd(n, d) == 1]
        seeds.sort(key=lambda n: canonical_hash(["COL-seed-order-v1", d, n]))
        for seed in seeds[:per_parameter]:
            records.append(admitted_record(d, seed))
    records.sort(key=lambda row: (row["certificate"]["d"], row["certificate"]["seed"]))
    audit = audit_records(records)
    audit.update({"max_parameter": max_d, "per_parameter_cap": per_parameter,
                  "unit_seeds_only": True, "holdout_roots_skipped": skipped,
                  "split_custody": "public deterministic partition; no secrecy attestation",
                  "training_tokens_claimed": 0,
                  "source_sha256": source_hashes()})
    return records, audit


def source_hashes() -> dict[str, str]:
    base = Path(__file__).parent
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in ("collatz_corpus.py", "collatz_verifier.py")}


def generate(out: Path, max_d: int = 1021, per_parameter: int = 8) -> dict:
    records, audit = generate_records(max_d, per_parameter)
    payloads = {}
    for split in ("train", "dev"):
        selected = [row for row in records if row["split"] == split]
        payloads[f"{split}.jsonl"] = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    payloads["manifest.json"] = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    for name, payload in payloads.items():
        if (out / name).exists() and (out / name).read_text() != payload:
            raise ValueError("existing evaluated export differs; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        if not (out / name).exists():
            with (out / name).open("x") as stream:
                stream.write(payload)
    return audit


def audit_directory(out: Path) -> dict:
    records = []
    for split in ("train", "dev"):
        for line in (out / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError("record placed in the wrong partition file")
            records.append(row)
    records.sort(key=lambda row: (row["certificate"]["d"], row["certificate"]["seed"]))
    result = audit_records(records)
    manifest = json.loads((out / "manifest.json").read_text())
    if any(manifest.get(k) != v for k, v in result.items()):
        raise ValueError("dataset manifest does not match independently replayed records")
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError("generator or verifier source differs from the export manifest")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "audit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-d", type=int, default=1021)
    parser.add_argument("--per-parameter", type=int, default=8)
    args = parser.parse_args()
    result = (generate(args.out, args.max_d, args.per_parameter)
              if args.command == "generate" else audit_directory(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
