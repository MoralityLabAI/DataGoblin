"""Audit native-observer replay accounting without authorizing deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from datagoblin.training_gym.molt_observer_allocation import map_bytes

METHODS = ("stack", "terminal")
GATE = 0.1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count(value: object) -> int:
    require(type(value) is int and value >= 0, "invalid nonnegative count")
    return value


def number(value: object) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid nonnegative measurement")
    return float(value)


def digest(value: object) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "missing content identity")
    return value


def audit(report: dict) -> dict:
    """Recompute payload totals from every declared donor, including fallbacks.

    Source hashes and teacher-replay acceptances are claims of the input
    report. This checks its internal contract, not the model or verifier.
    """
    require(report.get("schema") == "molt09-native-observer-v1"
            and report.get("status") == "engineering_development_only", "unexpected schema")
    require(report.get("promotion_authorized") is False
            and report.get("formal_certificate_authorized") is False, "unexpected authority")
    digest(report["teacher_checkpoint_sha256"])
    digest(report["native_factor_sha256"])
    c, k = count(report["native_span_rank"]), count(report["compact_rank"])
    require(1 <= k < c <= 16, "unexpected native chart")
    basis = count(report["common_basis_bytes"])
    width, rem = divmod(basis, 8 * c)
    require(rem == 0 and width >= c, "invalid common basis size")
    require(report.get("teacher_targets_shared_across_arms") is True, "missing shared target contract")
    maps = report["map_receipts"]
    donors = report["donor_receipts"]
    require(isinstance(maps, list) and 1 <= len(maps) <= 8, "invalid map roster")
    require(isinstance(donors, list) and len(donors) <= 512, "invalid donor roster")
    groups = defaultdict(dict)
    for r in donors:
        method, anchor, donor = r["method"], r["anchor"], r["donor"]
        require(method in METHODS and isinstance(anchor, str) and isinstance(donor, str)
                and anchor and donor and anchor != donor, "invalid donor identity")
        key = (anchor, method)
        require(donor not in groups[key], "duplicate donor receipt")
        delta = digest(r["delta_digest"])
        rel, absolute = number(r["relative_error"]), number(r["absolute_error"])
        parity, accepted = r["argmax_parity"], r["accepted"]
        require(type(parity) is bool and type(accepted) is bool, "invalid gate booleans")
        require(accepted == (parity and rel <= GATE), "teacher gate receipt inconsistent")
        groups[key][donor] = (delta, accepted, rel, absolute)
    anchors, bindings = set(), set()
    native_total = target_total = ambient_total = donor_count = 0
    planned = dict.fromkeys(METHODS, 0)
    selected_map_count = dict.fromkeys(METHODS, 0)
    mixed_total = 0
    detail = []
    for m in maps:
        anchor = m["anchor"]
        binding = digest(m["binding_digest"])
        require(isinstance(anchor, str) and anchor and anchor not in anchors
                and binding not in bindings, "duplicate anchor or binding")
        anchors.add(anchor)
        bindings.add(binding)
        n = count(m["donor_count"])
        require(2 <= n <= 32, "unexpected donor count")
        require(m["native_components"] == c and m["rank"] == k
                and m["common_basis_bytes"] == basis, "inconsistent chart accounting")
        require(number(m["native_api_parity_max_abs"]) <= 3e-4
                and number(m["clean_parity_max_abs"]) <= 3e-4
                and number(m["fd_relative_error"]) <= 0.005, "failed engineering parity gate")
        observer_rows = count(m["observer_rows"])
        require(observer_rows > 0, "empty observer")
        native, ambient, target = 8 * c * n, 8 * width * n, 4 * observer_rows * n
        require((native, ambient, target) == (m["native_coordinate_bytes"], m["ambient_write_bytes"],
                m["common_teacher_target_bytes"]), "incorrect shared-payload accounting")
        native_total += native
        ambient_total += ambient
        target_total += target
        donor_count += n
        reference = None
        options = [(native, "native")]
        detail_row = {"anchor": anchor, "binding_digest": binding, "donors": n, "methods": {}}
        require(set(m["methods"]) == set(METHODS), "missing method accounting")
        for method in METHODS:
            records = groups.get((anchor, method), {})
            require(len(records) == n, "missing donor receipt")
            ids = {d: r[0] for d, r in records.items()}
            if reference is None:
                reference = ids
            require(ids == reference, "methods use different donor displacements")
            distinct = len(set(ids.values()))
            require(distinct == m["distinct_displacement_count"], "distinct count mismatch")
            accepted = sum(r[1] for r in records.values())
            cost = map_bytes(c, k) + n + 8 * (k * accepted + c * (n - accepted))
            use = distinct == n and cost < native
            chosen = cost if use else native
            expected = {
                "accepted": accepted, "fallback": n - accepted, "map_bytes": map_bytes(c, k),
                "candidate_payload_bytes": cost, "planned_payload_bytes": chosen,
                "amortized_map_selected": use,
            }
            claim = m["methods"][method]
            require(all(claim[key] == value for key, value in expected.items()),
                    "native map cost or admission inconsistent")
            require(abs(number(claim["relative_error_max"]) - max(r[2] for r in records.values())) < 1e-12,
                    "relative error summary mismatch")
            planned[method] += chosen
            selected_map_count[method] += int(use)
            detail_row["methods"][method] = expected
            options.append((chosen, method))
        minimum, mode = min(options, key=lambda pair: (pair[0], pair[1] != "native", pair[1]))
        mixed_total += minimum
        detail_row["best_payload_mode"] = mode
        detail_row["best_payload_bytes"] = minimum
        detail.append(detail_row)
    require(set(groups) == {(a, m) for a in anchors for m in METHODS}, "unexpected donor group")
    require(donor_count == report["bank_records"], "bank record count mismatch")
    common = basis + target_total
    return {
        "schema": "molt09-native-accounting-v1",
        "status": "offline_internal_accounting_only",
        "groups": len(anchors),
        "records": donor_count,
        "width": width,
        "native_rank": c,
        "compressed_rank": k,
        "ambient_write_bytes": ambient_total,
        "common_basis_bytes": basis,
        "common_teacher_target_bytes": target_total,
        "native_coordinate_bytes": native_total,
        "native_total_with_common_payload": common + native_total,
        "methods": {m: {"map_groups": selected_map_count[m], "planned_bytes": planned[m],
                        "saved_vs_native": native_total - planned[m],
                        "total_with_common_payload": common + planned[m]} for m in METHODS},
        "best_mixed_total_with_common_payload": common + mixed_total,
        "best_mixed_saved_vs_native": native_total - mixed_total,
        "details": detail,
        "source_identity_verified": False,
        "hard_process_tree_cap_verified": False,
        "certificate_authorized": False,
        "promotion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.report.stat().st_size > 32 * 1024 * 1024:
        parser.error("report exceeds fixture cap")
    raw = args.report.read_bytes()
    result = audit(json.loads(raw))
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, sort_keys=True))


if __name__ == "__main__":
    main()
