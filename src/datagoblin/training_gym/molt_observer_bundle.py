"""MOLT-08 storage accounting for conditional observer-error receipts.

This is an offline, forecast-only allocator. A report's eligibility claims
are not independently verified numerical or logical certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from datagoblin.training_gym.molt_observer_allocation import MAX_DIMENSION, map_bytes

METHODS = ("global_curvature", *(f"interval{s}_{kind}"
           for s in (1, 4, 16) for kind in ("scalar", "signed")))


@dataclass(frozen=True)
class Record:
    id: str
    eligible_ranks: tuple[int, ...]
    bound_bytes: int

    def validate(self, dimension: int) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("nonempty record identity required")
        if type(self.bound_bytes) is not int or not 0 <= self.bound_bytes <= 65536:
            raise ValueError("invalid bound metadata size")
        if len(set(self.eligible_ranks)) != len(self.eligible_ranks) or any(
            type(k) is not int or not 1 <= k <= dimension for k in self.eligible_ranks
        ):
            raise ValueError("invalid rank set")


@dataclass(frozen=True)
class StoredRecord:
    id: str
    rank: int | None
    bytes: int


def plan_bundle(dimension: int, records: tuple[Record, ...]) -> dict:
    """Minimize map plus per-record payload, with exact ambient fallback.

    A single shared map contains all columns up to its chosen rank.
    Every record independently chooses its cheapest eligible prefix rank.
    All-ambient storage is an explicit zero-map alternative.
    """
    if type(dimension) is not int or not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError("dimension outside bounds")
    if not 1 <= len(records) <= 4096 or len({r.id for r in records}) != len(records):
        raise ValueError("unique nonempty record roster required")
    for r in records:
        r.validate(dimension)
    dense = 8 * dimension
    ordered = sorted(records, key=lambda r: r.id)
    best_rank = 0
    best_bytes = dense * len(records)
    best_records = tuple(StoredRecord(r.id, None, dense) for r in ordered)
    for max_rank in range(1, dimension + 1):
        chosen = []
        for r in ordered:
            allowed = [k for k in r.eligible_ranks if k <= max_rank]
            rank = min(allowed, default=dimension)
            payload = 8 * rank + r.bound_bytes
            chosen.append(StoredRecord(r.id, rank, payload)
                          if allowed and payload < dense else StoredRecord(r.id, None, dense))
        size = map_bytes(dimension, max_rank) + sum(r.bytes for r in chosen)
        # Prefer ambient or a smaller map when total payload is identical.
        if size < best_bytes:
            best_rank, best_bytes, best_records = max_rank, size, tuple(chosen)
    return {
        "dimension": dimension,
        "map_rank": best_rank,
        "map_bytes": map_bytes(dimension, best_rank) if best_rank else 0,
        "dense_bytes": dense * len(records),
        "planned_bytes": best_bytes,
        "payload_saved_bytes": dense * len(records) - best_bytes,
        "compressed_records": sum(r.rank is not None for r in best_records),
        "records": [asdict(r) for r in best_records],
        "conditional_eligibility_only": True,
        "certificate_authorized": False,
    }


def report_bundles(report: dict, method: str) -> dict:
    """Account for the exact development roster, without multiplying reuse.

    Grouping is by the fixed fixture's model seed and root. It is not the
    production map-binding key and cannot authorize a cache reuse.
    """
    if method not in METHODS:
        raise ValueError("conditional nonlinear method required")
    if (report.get("schema") != "molt08-observer-error-v1"
            or report.get("status") != "development_only"
            or report.get("conditional_real_arithmetic_only") is not True
            or report.get("formal_certificate_authorized") is not False):
        raise ValueError("unexpected report contract")
    rows = report.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 7680:
        raise ValueError("invalid candidate roster")
    groups: dict[tuple[int, str], dict[float, dict[int, bool]]] = {}
    metadata: dict[tuple[int, str], int] = {}
    for row in rows:
        seed, root, dose, rank = (row[k] for k in ("seed", "root", "dose", "rank"))
        if (type(seed) is not int or not isinstance(root, str) or not root
                or type(dose) not in (float, int) or dose not in (0.25, 0.5, 1.0)
                or type(rank) is not int or not 1 <= rank <= 5):
            raise ValueError("unexpected candidate key")
        accepted = row["accepts"][method]
        if type(accepted) is not bool or type(row["sign_parity"]) is not bool:
            raise ValueError("invalid eligibility receipt")
        if accepted and not row["sign_parity"]:
            raise ValueError("reported conditional gate failed its parity audit")
        key = (seed, root)
        per_dose = groups.setdefault(key, {}).setdefault(float(dose), {})
        if rank in per_dose:
            raise ValueError("duplicate rank receipt")
        per_dose[rank] = accepted
        approx = row["approx"]
        if not isinstance(approx, list) or len(approx) != 5:
            raise ValueError("fixture readout dimension changed")
        metadata[key] = 16 * len(approx) if method.endswith("_signed") else 8
    bundles = []
    for (seed, root), doses in sorted(groups.items()):
        if set(doses) != {0.25, 0.5, 1.0} or any(set(rs) != set(range(1, 6)) for rs in doses.values()):
            raise ValueError("incomplete fixed fixture roster")
        records = tuple(Record(str(d), tuple(k for k, ok in sorted(rs.items()) if ok), metadata[(seed, root)])
                        for d, rs in sorted(doses.items()))
        bundles.append({"seed": seed, "root": root, **plan_bundle(16, records)})
    return {
        "schema": "molt08-bundle-accounting-v1",
        "status": "offline_development_accounting",
        "method": method,
        "groups": len(bundles),
        "records": sum(len(b["records"]) for b in bundles),
        "dense_bytes": sum(b["dense_bytes"] for b in bundles),
        "planned_bytes": sum(b["planned_bytes"] for b in bundles),
        "payload_saved_bytes": sum(b["payload_saved_bytes"] for b in bundles),
        "map_groups": sum(b["map_rank"] > 0 for b in bundles),
        "bundles": bundles,
        "production_binding_verified": False,
        "hard_process_tree_cap_verified": False,
        "certificate_authorized": False,
        "promotion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.report.stat().st_size > 32 * 1024 * 1024:
        parser.error("report exceeds fixture size cap")
    raw = args.report.read_bytes()
    result = report_bundles(json.loads(raw), args.method)
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "bundles"}, sort_keys=True))


if __name__ == "__main__":
    main()
