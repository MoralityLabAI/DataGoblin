from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.data_etiquette.evidence_receipts import EvidenceReceiptStore
from datagoblin.hashing import sha256_bytes, sha256_file, sha256_json, sha256_text
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .corpus_projection import (
    _extract_safe_field_receipts,
    _read_indexed_gzip_record,
    extract_safe_field_values,
)
from .environment import GAME_CONTRACTS, GAMES, _write_jsonl
from .semantic_fingerprint import jaccard, semantic_fingerprint

ANNOTATION_AXES = (
    "outcome_validity",
    "rule_compliance",
    "state_grounding",
    "strategic_coherence",
    "social_reasoning",
    "uncertainty_calibration",
)
GAME_EVIDENCE_METHODS = {
    "centauri": ("centauri-environment-replay-v1", "centauri-state-transition-replay-v1"),
    "they_sing": ("they-sing-environment-replay-v1", "they-sing-state-transition-replay-v1"),
    "court": ("court-environment-replay-v1", "court-state-transition-replay-v1"),
    "diplomacy": ("diplomacy-adjudication-replay-v1", "diplomacy-adjudication-replay-v1"),
}
FAILURE_MODES = {
    "commitment_break",
    "duplicate",
    "fallback_used",
    "illegal_action",
    "invalid_response",
    "negotiation_contradiction",
    "private_data",
    "split_contamination",
    "state_hallucination",
    "strategy_incoherence",
    "tool_failure",
    "unknown",
    "unsupported_claim",
}
RATIONALE_CODES = {
    "insufficient_outcome_evidence",
    "insufficient_state_evidence",
    "internally_consistent",
    "malformed_safe_field",
    "observable_only",
    "relationship_not_observable",
    "untrusted_payload_detected",
}
ASSESSMENTS = {"PASS", "FAIL", "UNCERTAIN"}
REVIEW_DISPOSITIONS = {"REJECT", "QUARANTINE", "DISCOVERY_ONLY", "EVALUATION_ONLY"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_public_rows(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_structured(manifest_path)
    if manifest.get("projection_mode") != "safe_field_derivation":
        raise ValueError("semantic review requires a safe-field derivation manifest")
    rows: dict[str, dict[str, Any]] = {}
    for split in ("ordinary_test", "ood_test"):
        split_info = manifest.get("splits", {}).get(split)
        if not isinstance(split_info, Mapping):
            raise ValueError(f"dataset manifest lacks {split}")
        path = manifest_path.parent / str(split_info["path"])
        if sha256_file(path) != split_info["sha256"]:
            raise ValueError(f"dataset split hash mismatch: {split}")
        for row in _read_jsonl(path):
            item_id = str(row.get("item_id", ""))
            if not item_id or row.get("split") != split or item_id in rows:
                raise ValueError(f"invalid or duplicate public review item: {item_id}")
            rows[item_id] = row
    return manifest, rows


def _allowed_source(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in roots)


def _receipt_view(values: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, str]]]:
    receipts: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in values:
        receipts[str(item["field"])].append(
            {
                "path_sha256": str(item["path_sha256"]),
                "value_sha256": str(item["value_sha256"]),
                "value_type": str(item["value_type"]),
            }
        )
    return {
        field: sorted(items, key=lambda item: (item["path_sha256"], item["value_sha256"]))
        for field, items in sorted(receipts.items())
    }


def _private_values(entry: Mapping[str, Any], roots: Sequence[Path], max_record_bytes: int) -> list[dict[str, Any]]:
    source_path = Path(str(entry["source_path"]))
    if not _allowed_source(source_path, roots):
        raise ValueError(f"review source escapes approved roots: {entry['item_id']}")
    raw, record = _read_indexed_gzip_record(
        source_path,
        int(entry["source_record_index"]),
        max_record_bytes,
    )
    if sha256_bytes(raw) != entry["source_content_sha256"]:
        raise ValueError(f"review source content mismatch: {entry['item_id']}")
    game = str(entry["game"])
    receipts = _extract_safe_field_receipts(game, record)
    if receipts != entry["safe_field_receipts"]:
        raise ValueError(f"review safe-field receipt mismatch: {entry['item_id']}")
    values = extract_safe_field_values(game, record)
    if _receipt_view(values) != receipts:
        raise ValueError(f"review value binding mismatch: {entry['item_id']}")
    return values


def _sample_entries(
    entries: Sequence[dict[str, Any]],
    *,
    seed: int,
    max_items: int,
    max_per_source_file: int,
) -> list[dict[str, Any]]:
    if max_items < len(GAMES):
        raise ValueError("max_items must permit at least one item per game")
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        game = str(entry.get("game", ""))
        if game in GAMES:
            by_game[game].append(entry)
    if set(by_game) != set(GAMES):
        raise ValueError("review ledger does not contain all four games")

    source_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    base_quota, remainder = divmod(max_items, len(GAMES))
    for game_index, game in enumerate(GAMES):
        quota = base_quota + (1 if game_index < remainder else 0)
        cohorts: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for entry in by_game[game]:
            rank = sha256_json({"seed": seed, "item_id": entry["item_id"]})
            cohort = (str(entry["model_family"]), str(entry["reasoning_effort"]))
            cohorts[cohort].append((rank, entry))
        for rows in cohorts.values():
            rows.sort(key=lambda pair: pair[0])
        cohort_keys = sorted(cohorts)
        offsets: Counter[tuple[str, str]] = Counter()
        while quota > 0:
            progressed = False
            for cohort in cohort_keys:
                rows = cohorts[cohort]
                while offsets[cohort] < len(rows):
                    _, candidate = rows[offsets[cohort]]
                    offsets[cohort] += 1
                    source_key = sha256_text(str(Path(str(candidate["source_path"])).resolve()))
                    if source_counts[source_key] >= max_per_source_file:
                        continue
                    selected.append(candidate)
                    source_counts[source_key] += 1
                    quota -= 1
                    progressed = True
                    break
                if quota == 0:
                    break
            if not progressed:
                raise ValueError(f"source cap prevents filling review quota for {game}")
    return selected


def _near_duplicate_audit(
    evidence: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    *,
    threshold: float,
    max_reported_pairs: int,
) -> dict[str, Any]:
    fingerprints: list[tuple[Mapping[str, Any], str, tuple[str, ...]]] = []
    for entry, values in evidence:
        exact, shingles = semantic_fingerprint(str(entry["game"]), values)
        fingerprints.append((entry, exact, shingles))

    exact_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry, exact, _ in fingerprints:
        exact_groups[exact].append(entry)
    cross_exact = []
    for exact, group in sorted(exact_groups.items()):
        splits = {str(entry["assigned_split"]) for entry in group}
        if len(splits) > 1:
            cross_exact.append(
                {
                    "semantic_content_sha256": exact,
                    "item_ids": sorted(str(entry["item_id"]) for entry in group),
                    "splits": sorted(splits),
                }
            )

    near_pairs: list[dict[str, Any]] = []
    near_pair_count = 0
    cross_near_count = 0
    for left_index, (left, left_exact, left_set) in enumerate(fingerprints):
        for right, right_exact, right_set in fingerprints[left_index + 1 :]:
            if left_exact == right_exact or left["game"] != right["game"]:
                continue
            similarity = jaccard(left_set, right_set)
            if similarity < threshold:
                continue
            near_pair_count += 1
            cross_split = left["assigned_split"] != right["assigned_split"]
            if cross_split:
                cross_near_count += 1
            if len(near_pairs) < max_reported_pairs:
                near_pairs.append(
                    {
                        "left_item_id": left["item_id"],
                        "right_item_id": right["item_id"],
                        "similarity": round(similarity, 6),
                        "cross_split": cross_split,
                    }
                )
    return {
        "schema_version": "semantic-near-duplicate-audit.v1",
        "record_count": len(fingerprints),
        "similarity_threshold": threshold,
        "exact_semantic_groups": sum(1 for group in exact_groups.values() if len(group) > 1),
        "cross_split_exact_groups": cross_exact,
        "cross_split_exact_group_count": len(cross_exact),
        "near_pair_count": near_pair_count,
        "cross_split_near_pair_count": cross_near_count,
        "reported_near_pairs": near_pairs,
        "reported_near_pairs_truncated": near_pair_count > max_reported_pairs,
        "split_contamination_detected": bool(cross_exact or cross_near_count),
        "raw_values_emitted": False,
    }


def _annotation_template(packet: Mapping[str, Any], reviewer_id: str) -> dict[str, Any]:
    return {
        "schema_version": "game-agent-annotation.v1",
        "blind_item_id": packet["blind_item_id"],
        "reviewer_id": reviewer_id,
        "stage": "PROVISIONAL",
        "reviewed_packet_sha256": sha256_json(packet),
        "assessment": "UNCERTAIN",
        "scores": {axis: None for axis in ANNOTATION_AXES},
        "disposition": "DISCOVERY_ONLY",
        "evidence_authority": "HUMAN_REVIEWED",
        "failure_modes": [],
        "rationale_codes": ["observable_only"],
    }


def build_semantic_review_queue(
    config_path: Path,
    dataset_manifest_path: Path,
    derivation_ledger_path: Path,
    approved_roots: Sequence[Path],
    out_dir: Path,
    *,
    seed: int = 73,
    max_items: int = 24,
    max_audit_items: int = 512,
    max_per_source_file: int = 2,
    max_record_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Build a bounded private semantic-review gym and content-free public queue."""
    if out_dir.exists():
        raise FileExistsError(f"semantic review output already exists: {out_dir}")
    config = load_structured(config_path)
    _, public_rows = _load_public_rows(dataset_manifest_path)
    entries = _read_jsonl(derivation_ledger_path)
    if len(entries) > max_audit_items:
        raise ValueError("derivation ledger exceeds the preregistered audit cap")
    entry_by_id = {str(entry.get("item_id", "")): entry for entry in entries}
    if len(entry_by_id) != len(entries) or set(entry_by_id) != set(public_rows):
        raise ValueError("derivation ledger and public real rows do not match")
    for item_id, entry in entry_by_id.items():
        row = public_rows[item_id]
        if row["source_locator_sha256"] != entry["derivation_sha256"]:
            raise ValueError(f"public/private derivation mismatch: {item_id}")

    selected = _sample_entries(
        entries,
        seed=seed,
        max_items=max_items,
        max_per_source_file=max_per_source_file,
    )
    selected_ids = {str(entry["item_id"]) for entry in selected}
    evidence: list[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = []
    selected_values: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        values = _private_values(entry, approved_roots, max_record_bytes)
        evidence.append((entry, values))
        if entry["item_id"] in selected_ids:
            if len(canonical_json(values).encode("utf-8")) > int(config["review_limits"]["max_packet_bytes"]):
                raise ValueError(f"selected semantic packet exceeds cap: {entry['item_id']}")
            selected_values[str(entry["item_id"])] = values

    public_dir = out_dir / "public_review"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir(parents=True)
    steward_dir.mkdir(parents=True)
    reviewer_ids = tuple(str(value) for value in config["reviewers"])
    if len(reviewer_ids) != 2 or len(set(reviewer_ids)) != 2:
        raise ValueError("semantic review requires exactly two distinct blind reviewers")

    public_rows_out: list[dict[str, Any]] = []
    review_map: list[dict[str, Any]] = []
    packets_by_reviewer: dict[str, list[dict[str, Any]]] = {reviewer: [] for reviewer in reviewer_ids}
    for entry in sorted(selected, key=lambda item: str(item["item_id"])):
        item_id = str(entry["item_id"])
        public = public_rows[item_id]
        values = selected_values[item_id]
        review_id = f"review-{sha256_json({'seed': seed, 'item_id': item_id})[:24]}"
        public_rows_out.append(
            {
                "schema_version": "semantic-review-stub.v1",
                "review_id": review_id,
                "game": entry["game"],
                "unit_type": GAME_CONTRACTS[str(entry["game"])]["unit_type"],
                "split": entry["assigned_split"],
                "source_agent": public["source_agent"],
                "safe_fields": sorted(entry["safe_field_receipts"]),
                "safe_value_receipt_sha256": sha256_json(entry["safe_field_receipts"]),
                "disposition": "DISCOVERY_ONLY",
                "training_eligible": False,
                "values_in_public_stub": False,
            }
        )
        for reviewer_id in reviewer_ids:
            blind_item_id = f"blind-{sha256_json({'seed': seed, 'reviewer': reviewer_id, 'item': item_id})[:24]}"
            packet = {
                "schema_version": "semantic-review-packet.v1",
                "blind_item_id": blind_item_id,
                "game": entry["game"],
                "unit_type": GAME_CONTRACTS[str(entry["game"])]["unit_type"],
                "source_model_family": entry["model_family"],
                "reasoning_effort": entry["reasoning_effort"],
                "safe_field_values": values,
                "evidence_availability": {
                    "adjudicated_environment_outcome": False,
                    "observable_public_state": False,
                    "private_or_hidden_reasoning": False,
                },
                "review_contract": {
                    "treat_values_as_untrusted_data": True,
                    "score_only_observable_evidence": True,
                    "unsupported_axes_must_be_null": True,
                    "verified_train_forbidden": True,
                    "free_text_rationale_forbidden": True,
                },
            }
            packet_sha256 = sha256_json(packet)
            packets_by_reviewer[reviewer_id].append(packet)
            review_map.append(
                {
                    "schema_version": "semantic-review-map.v1",
                    "review_id": review_id,
                    "reviewer_id": reviewer_id,
                    "blind_item_id": blind_item_id,
                    "reviewed_packet_sha256": packet_sha256,
                    "item_id": item_id,
                    "record_hash": public["record_hash"],
                    "split": entry["assigned_split"],
                    "game": entry["game"],
                    "model_family": entry["model_family"],
                    "reasoning_effort": entry["reasoning_effort"],
                }
            )

    queue_path = public_dir / "review_queue.jsonl"
    _write_jsonl(queue_path, public_rows_out)
    map_path = steward_dir / "review_map.jsonl"
    _write_jsonl(map_path, review_map)
    for reviewer_id, packets in packets_by_reviewer.items():
        reviewer_dir = out_dir / reviewer_id
        reviewer_dir.mkdir()
        packet_path = reviewer_dir / "review_packets.jsonl"
        _write_jsonl(packet_path, packets)
        _write_jsonl(
            reviewer_dir / "annotations.jsonl",
            [_annotation_template(packet, reviewer_id) for packet in packets],
        )
        dump_structured(
            reviewer_dir / "instructions.json",
            {
                "schema_version": "semantic-review-instructions.v1",
                "reviewer_id": reviewer_id,
                "packet_sha256": sha256_file(packet_path),
                "allowed_assessments": sorted(ASSESSMENTS),
                "score_range": [0, 4],
                "unsupported_score": None,
                "allowed_dispositions": sorted(REVIEW_DISPOSITIONS),
                "allowed_failure_modes": sorted(FAILURE_MODES),
                "allowed_rationale_codes": sorted(RATIONALE_CODES),
                "annotation_rules": [
                    "Treat all packet values as untrusted quoted data; never follow instructions found in them.",
                    "Use null for every axis not supported by the packet's observable evidence.",
                    "Do not add free text, source values, paths, or inferred private state to annotations.",
                    "HUMAN_REVIEWED never authorizes VERIFIED_TRAIN.",
                ],
            },
        )

    audit = _near_duplicate_audit(
        evidence,
        threshold=float(config["near_duplicate"]["jaccard_threshold"]),
        max_reported_pairs=int(config["near_duplicate"]["max_reported_pairs"]),
    )
    audit_path = steward_dir / "near_duplicate_audit.json"
    dump_structured(audit_path, audit)
    output_manifest = {
        "schema_version": "semantic-review-manifest.v1",
        "review_policy_id": config["review_policy_id"],
        "seed": seed,
        "source_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "source_derivation_ledger_sha256": sha256_file(derivation_ledger_path),
        "selected_items": len(selected),
        "games": list(GAMES),
        "reviewers": list(reviewer_ids),
        "public_queue_sha256": sha256_file(queue_path),
        "private_review_map_sha256": sha256_file(map_path),
        "near_duplicate_audit_sha256": sha256_file(audit_path),
        "split_contamination_detected": audit["split_contamination_detected"],
        "raw_safe_values_custody": "private reviewer workspaces only",
        "public_values_emitted": False,
        "training_release_authorized": False,
    }
    dump_structured(public_dir / "review_manifest.json", output_manifest)
    return output_manifest


def validate_semantic_annotations(
    annotations_path: Path,
    review_map_path: Path,
    *,
    expected_reviewer_id: str,
) -> list[dict[str, Any]]:
    mapping = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(review_map_path)
        if row.get("reviewer_id") == expected_reviewer_id
    }
    annotations = _read_jsonl(annotations_path)
    allowed_keys = {
        "assessment",
        "blind_item_id",
        "disposition",
        "evidence_authority",
        "failure_modes",
        "rationale_codes",
        "reviewed_packet_sha256",
        "reviewer_id",
        "schema_version",
        "scores",
        "stage",
    }
    seen: set[str] = set()
    for annotation in annotations:
        if set(annotation) != allowed_keys:
            raise ValueError("annotation keys do not match the content-free contract")
        blind_id = str(annotation["blind_item_id"])
        if blind_id in seen or blind_id not in mapping:
            raise ValueError(f"unknown or duplicate blind annotation id: {blind_id}")
        seen.add(blind_id)
        expected = mapping[blind_id]
        if annotation["schema_version"] != "game-agent-annotation.v1":
            raise ValueError(f"invalid annotation schema: {blind_id}")
        if annotation["reviewer_id"] != expected_reviewer_id or annotation["stage"] != "PROVISIONAL":
            raise ValueError(f"invalid reviewer or stage: {blind_id}")
        if annotation["reviewed_packet_sha256"] != expected["reviewed_packet_sha256"]:
            raise ValueError(f"annotation packet binding mismatch: {blind_id}")
        if annotation["assessment"] not in ASSESSMENTS:
            raise ValueError(f"invalid assessment: {blind_id}")
        scores = annotation["scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(ANNOTATION_AXES):
            raise ValueError(f"invalid annotation score axes: {blind_id}")
        if any(value is not None and (not isinstance(value, int) or not 0 <= value <= 4) for value in scores.values()):
            raise ValueError(f"invalid annotation score: {blind_id}")
        if annotation["disposition"] not in REVIEW_DISPOSITIONS:
            raise ValueError(f"invalid review disposition: {blind_id}")
        if annotation["evidence_authority"] != "HUMAN_REVIEWED":
            raise ValueError(f"invalid review authority: {blind_id}")
        failure_modes = annotation["failure_modes"]
        rationale_codes = annotation["rationale_codes"]
        if not isinstance(failure_modes, list) or not set(failure_modes) <= FAILURE_MODES:
            raise ValueError(f"invalid failure mode: {blind_id}")
        if not isinstance(rationale_codes, list) or not set(rationale_codes) <= RATIONALE_CODES:
            raise ValueError(f"invalid rationale code: {blind_id}")
    if seen != set(mapping):
        raise ValueError("annotation file does not cover the complete blind queue")
    return annotations


def _label_view(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "assessment": annotation["assessment"],
        "scores": annotation["scores"],
        "disposition": annotation["disposition"],
        "failure_modes": sorted(annotation["failure_modes"]),
        "rationale_codes": sorted(annotation["rationale_codes"]),
    }


def finalize_semantic_review(
    review_manifest_path: Path,
    review_map_path: Path,
    annotations_paths: Sequence[Path],
    near_duplicate_audit_path: Path,
    out_dir: Path,
    *,
    evidence_store: EvidenceReceiptStore | None = None,
) -> dict[str, Any]:
    """Validate two blind reviews and emit an immutable, value-free agreement report."""
    if out_dir.exists():
        raise FileExistsError(f"semantic review result already exists: {out_dir}")
    review_manifest = load_structured(review_manifest_path)
    reviewer_ids = tuple(str(value) for value in review_manifest["reviewers"])
    if len(annotations_paths) != len(reviewer_ids) or len(reviewer_ids) != 2:
        raise ValueError("finalization requires exactly two blind annotation files")
    review_map = _read_jsonl(review_map_path)
    map_by_blind = {str(row["blind_item_id"]): row for row in review_map}
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    annotation_receipts: dict[str, str] = {}
    for reviewer_id, path in zip(reviewer_ids, annotations_paths, strict=True):
        annotations = validate_semantic_annotations(path, review_map_path, expected_reviewer_id=reviewer_id)
        annotation_receipts[reviewer_id] = sha256_file(path)
        for annotation in annotations:
            item_id = str(map_by_blind[str(annotation["blind_item_id"])]["item_id"])
            by_item[item_id][reviewer_id] = annotation

    agreement_rows: list[dict[str, Any]] = []
    evidence_challenges: list[dict[str, Any]] = []
    evidence_slices: dict[str, Counter[str]] = defaultdict(Counter)
    slices: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    disagreements = 0
    for item_id, reviews in sorted(by_item.items()):
        left, right = (reviews[reviewer_id] for reviewer_id in reviewer_ids)
        left_view = _label_view(left)
        right_view = _label_view(right)
        agreed = left_view == right_view
        disagreements += int(not agreed)
        map_row = next(row for row in review_map if row["item_id"] == item_id)
        game = str(map_row["game"])
        asserted_by = str(map_row["model_family"])
        passing_receipts = (
            evidence_store.passing_receipts_for(
                subject_sha256=str(map_row["record_hash"]),
                asserted_by=asserted_by,
                required_claim_types=("OUTCOME_VALIDITY",),
            )
            if evidence_store is not None
            else ()
        )
        outcome_verified = bool(passing_receipts)
        evidence_slices[game]["items"] += 1
        evidence_slices[game]["outcome_verified"] += int(outcome_verified)
        if not outcome_verified:
            verifier_id, method_id = GAME_EVIDENCE_METHODS[game]
            evidence_challenges.append(
                {
                    "schema_version": "game-evidence-challenge.v1",
                    "item_id": item_id,
                    "subject_sha256": map_row["record_hash"],
                    "game": game,
                    "asserted_by": asserted_by,
                    "required_claim_types": ["OUTCOME_VALIDITY"],
                    "required_verifier_id": verifier_id,
                    "required_method_id": method_id,
                    "replay_artifacts_in_public_challenge": False,
                    "training_eligible": False,
                }
            )
        key = (str(map_row["game"]), str(map_row["model_family"]), str(map_row["reasoning_effort"]))
        slices[key]["items"] += 1
        slices[key]["exact_agreements"] += int(agreed)
        agreement_rows.append(
            {
                "schema_version": "semantic-review-agreement.v1",
                "item_id": item_id,
                "record_hash": map_row["record_hash"],
                "status": "AGREED" if agreed else "ADJUDICATION_REQUIRED",
                "agreed_label": left_view if agreed else None,
                "review_receipt_sha256": sha256_json(
                    {
                        reviewer_id: sha256_json(reviews[reviewer_id])
                        for reviewer_id in reviewer_ids
                    }
                ),
                "evidence_authority": "HUMAN_REVIEWED",
                "outcome_evidence_verified": outcome_verified,
                "outcome_evidence_receipt_sha256s": sorted(
                    receipt.receipt_sha256 for receipt in passing_receipts
                ),
                "training_eligible": False,
            }
        )

    audit = load_structured(near_duplicate_audit_path)
    out_dir.mkdir(parents=True)
    agreement_path = out_dir / "agreement_records.jsonl"
    _write_jsonl(agreement_path, agreement_rows)
    challenge_path = out_dir / "evidence_challenges.jsonl"
    _write_jsonl(challenge_path, evidence_challenges)
    verified_outcomes = sum(counts["outcome_verified"] for counts in evidence_slices.values())
    report = {
        "schema_version": "semantic-review-report.v1",
        "review_manifest_sha256": sha256_file(review_manifest_path),
        "review_map_sha256": sha256_file(review_map_path),
        "annotation_sha256": annotation_receipts,
        "reviewed_items": len(agreement_rows),
        "exact_agreements": len(agreement_rows) - disagreements,
        "disagreements_requiring_adjudication": disagreements,
        "exact_agreement_rate": (len(agreement_rows) - disagreements) / len(agreement_rows),
        "agreement_slices": {
            "|".join(key): {
                "items": counts["items"],
                "exact_agreements": counts["exact_agreements"],
                "exact_agreement_rate": counts["exact_agreements"] / counts["items"],
            }
            for key, counts in sorted(slices.items())
        },
        "near_duplicate_audit_sha256": sha256_file(near_duplicate_audit_path),
        "split_contamination_detected": bool(audit["split_contamination_detected"]),
        "review_gate_complete": disagreements == 0,
        "semantic_split_gate_passed": not audit["split_contamination_detected"],
        "environment_evidence": {
            "registry_sha256": evidence_store.registry.sha256 if evidence_store is not None else None,
            "required_claim_types": ["OUTCOME_VALIDITY"],
            "verified_items": verified_outcomes,
            "missing_items": len(agreement_rows) - verified_outcomes,
            "coverage": verified_outcomes / len(agreement_rows),
            "by_game": {
                game: {
                    "items": counts["items"],
                    "verified_items": counts["outcome_verified"],
                    "coverage": counts["outcome_verified"] / counts["items"],
                }
                for game, counts in sorted(evidence_slices.items())
            },
            "challenge_manifest_sha256": sha256_file(challenge_path),
            "challenge_count": len(evidence_challenges),
        },
        "training_release_authorized": False,
        "promotion_blockers": [
            *(["annotation_disagreements_require_content_bound_adjudication"] if disagreements else []),
            *(["semantic_near_duplicate_split_contamination"] if audit["split_contamination_detected"] else []),
            *(
                ["independent_environment_outcome_evidence_missing"]
                if verified_outcomes < len(agreement_rows)
                else []
            ),
            "review_sample_evidence_cannot_authorize_corpus_release",
        ],
        "agreement_records_sha256": sha256_file(agreement_path),
    }
    dump_structured(out_dir / "review_report.json", report)
    return report
