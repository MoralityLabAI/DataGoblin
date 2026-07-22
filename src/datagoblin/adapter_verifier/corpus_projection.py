from __future__ import annotations

import gzip
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_bytes, sha256_file, sha256_json, sha256_text
from datagoblin.serialization import dump_structured, load_structured

from .environment import GAME_CONTRACTS, GAMES, SPLITS, _load_config, _write_jsonl

MODEL_FAMILIES = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
REASONING_EFFORTS = ("medium", "high", "xhigh")
GAME_ALIASES = {
    "centauri": "centauri",
    "court": "court",
    "diplomacy": "diplomacy",
    "they sing": "they_sing",
    "they_sing": "they_sing",
    "theysing": "they_sing",
}
HIDDEN_REASONING_TOKENS = {
    "chain_of_thought",
    "content",
    "hidden_reasoning",
    "instructions",
    "prompt",
    "rationale",
    "rationale_summary",
    "reasoning",
    "response",
    "session",
    "session_id",
    "session_name",
    "strategic_plan",
    "text",
}
PRIVATE_TOKENS = {
    "credential",
    "hidden_win_condition",
    "password",
    "private_data",
    "private_memory",
    "private_messages",
    "private_press",
    "secret",
    "secrets",
    "your_private_state",
}
SAFE_METADATA_TOKENS = {
    "adjudicated_outcome",
    "chosen_action",
    "environment_outcome",
    "legal_actions",
    "orders",
    "phase",
    "public_action",
    "public_message",
    "public_observation",
    "public_press",
    "public_state",
    "source_checks",
    "state_delta",
    "state_hash",
    "target",
    "usage_bucket",
}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
SAFE_DERIVATION_PATHS: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "centauri": {
        "legal_actions": (("legal_actions",),),
    },
    "they_sing": {
        "legal_actions": (("legal_actions",),),
        "orders": (("target", "action", "orders"), ("target", "allocation", "orders")),
        "phase": (("phase",),),
    },
    "court": {
        "legal_actions": (("legal_actions",),),
        "phase": (("phase",),),
    },
    "diplomacy": {
        "orders": (("trace", "adjudicated_outcomes", "*", "orders"),),
        "source_checks": (("source_checks",),),
    },
}


def _token(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _known_tokens(config: Mapping[str, Any]) -> set[str]:
    configured = {_token(str(value)) for value in config["forbidden_fields"]}
    return configured | HIDDEN_REASONING_TOKENS | PRIVATE_TOKENS | SAFE_METADATA_TOKENS


def _shape_tokens(value: Any, known: set[str]) -> tuple[set[str], int, set[str]]:
    visible: set[str] = set()
    types: set[str] = set()
    unknown = 0
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            types.add("object")
            for key, nested in current.items():
                key_text = str(key)
                normalized = _token(key_text) if SAFE_IDENTIFIER.fullmatch(key_text) else ""
                if normalized in known:
                    visible.add(normalized)
                else:
                    unknown += 1
                stack.append(nested)
        elif isinstance(current, list):
            types.add("array")
            stack.extend(current[:32])
        elif current is None:
            types.add("null")
        elif isinstance(current, bool):
            types.add("boolean")
        elif isinstance(current, (int, float)):
            types.add("number")
        elif isinstance(current, str):
            types.add("string")
    return visible, unknown, types


def _normalize_game(value: Any, fallback: str | None = None) -> str | None:
    if isinstance(value, str):
        normalized = re.sub(r"[^a-z]+", " ", value.lower()).strip()
        compact = normalized.replace(" ", "_")
        for candidate in (normalized, compact, normalized.replace(" ", "")):
            if candidate in GAME_ALIASES:
                return GAME_ALIASES[candidate]
    return fallback if fallback in GAMES else None


def _allowed_cohorts(receipt: Mapping[str, Any]) -> set[tuple[str, str]]:
    if receipt.get("accepted") is not True:
        raise ValueError("AICourt receipt is not accepted")
    by_cell = receipt.get("tokens", {}).get("by_cell", {})
    if not isinstance(by_cell, Mapping):
        raise ValueError("AICourt receipt lacks aggregate token cells")
    allowed: set[tuple[str, str]] = set()
    for cell in by_cell:
        parts = str(cell).split(":")
        if len(parts) >= 2 and parts[0] in MODEL_FAMILIES and parts[1] in REASONING_EFFORTS:
            allowed.add((parts[0], parts[1]))
    if not allowed:
        raise ValueError("AICourt receipt does not declare any supported model/effort cohorts")
    return allowed


def _iter_gzip_records(
    path: Path, max_records: int, max_record_bytes: int
) -> Iterable[tuple[int, bytes, Any]]:
    with gzip.open(path, "rb") as handle:
        for record_index in range(1, max_records + 1):
            raw = handle.readline(max_record_bytes + 1)
            if not raw:
                return
            if len(raw) > max_record_bytes or not raw.endswith(b"\n"):
                return
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping):
                yield record_index, raw, value


def _expected(game: str, keys: set[str], schema_supported: bool, unknown_count: int) -> dict[str, Any]:
    if not schema_supported:
        return {
            "game": game,
            "unit_type": GAME_CONTRACTS[game]["unit_type"],
            "mapped_fields": [],
            "disposition": "QUARANTINE",
            "evidence_authority": "UNKNOWN",
            "should_parse": False,
            "should_abstain": True,
            "failure_modes": ["unknown"],
        }
    hidden = bool(keys & HIDDEN_REASONING_TOKENS)
    private = bool(keys & PRIVATE_TOKENS)
    if hidden:
        failures = ["hidden_reasoning"]
        if private:
            failures.append("private_data")
        return {
            "game": game,
            "unit_type": GAME_CONTRACTS[game]["unit_type"],
            "mapped_fields": [],
            "disposition": "REJECT",
            "evidence_authority": "UNKNOWN",
            "should_parse": True,
            "should_abstain": True,
            "failure_modes": failures,
        }
    if private:
        return {
            "game": game,
            "unit_type": GAME_CONTRACTS[game]["unit_type"],
            "mapped_fields": [],
            "disposition": "QUARANTINE",
            "evidence_authority": "UNKNOWN",
            "should_parse": True,
            "should_abstain": True,
            "failure_modes": ["private_data"],
        }
    if unknown_count:
        return {
            "game": game,
            "unit_type": GAME_CONTRACTS[game]["unit_type"],
            "mapped_fields": [],
            "disposition": "QUARANTINE",
            "evidence_authority": "UNKNOWN",
            "should_parse": False,
            "should_abstain": True,
            "failure_modes": ["unknown"],
        }
    return {
        "game": game,
        "unit_type": GAME_CONTRACTS[game]["unit_type"],
        "mapped_fields": sorted(keys & set(GAME_CONTRACTS[game]["safe_fields"])),
        "disposition": "DISCOVERY_ONLY",
        "evidence_authority": "UNKNOWN",
        "should_parse": True,
        "should_abstain": False,
        "failure_modes": ["unsupported_claim"],
    }


def _public_row(
    raw: bytes,
    value: Mapping[str, Any],
    config: Mapping[str, Any],
    fallback_game: str | None,
    allowed_cohorts: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    game = _normalize_game(value.get("env"), fallback_game)
    if game is None:
        return None
    model = value.get("model") if value.get("model") in MODEL_FAMILIES else "unresolved"
    effort = value.get("reasoning_effort") if value.get("reasoning_effort") in REASONING_EFFORTS else "unknown"
    if game != "diplomacy" and (model, effort) not in allowed_cohorts:
        return None
    keys, unknown_count, types = _shape_tokens(value, _known_tokens(config))
    schema_supported = bool(keys)
    digest = sha256_text(raw.hex())
    split = "ordinary_test" if int(digest[:2], 16) < 192 else "ood_test"
    source_agent = {
        "model_family": model,
        "reasoning_effort": effort,
        "metadata_authority": (
            "receipt_bound_manifest_claim" if game != "diplomacy" else "unresolved_manifest_claim"
        ),
    }
    public_input = {
        "schema_version": "game-adapter-input.v1",
        "item_id": f"corpus-{digest[:24]}",
        "split": split,
        "game_hint": game,
        "source_schema": f"{game}-content-free-shape-v1",
        "source_locator_sha256": digest,
        "source_agent": source_agent,
        "projection_origin": "approved_export_shape",
        "source_shape": {
            "keys": sorted(keys),
            "unknown_field_count": unknown_count,
            "value_types": sorted(types),
            "record_kind": GAME_CONTRACTS[game]["unit_type"],
            "visibility": "actor_private" if keys & PRIVATE_TOKENS else "public",
            "flags": {
                "aggregate_receipt_only": True,
                "environment_verified": False,
                "evaluation_material": False,
                "explicit_hidden_reasoning": bool(keys & HIDDEN_REASONING_TOKENS),
                "private_visibility": bool(keys & PRIVATE_TOKENS),
                "unresolved_fields": bool(unknown_count),
                "schema_supported": schema_supported,
                "sealed_evaluation": False,
            },
        },
    }
    return public_input, _expected(game, keys, schema_supported, unknown_count)


def _sealed_canary(row: Mapping[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    game = str(row["game_hint"])
    agent = row["source_agent"]
    digest = sha256_json({"seed": seed, "game": game, "source_agent": agent, "kind": "sealed_canary"})
    canary = {
        **{key: value for key, value in row.items() if key != "record_hash"},
        "item_id": f"canary-{digest[:24]}",
        "split": "sealed_poison",
        "source_locator_sha256": digest,
        "projection_origin": "environment_canary",
        "source_shape": {
            **row["source_shape"],
            "flags": {
                **row["source_shape"]["flags"],
                "evaluation_material": True,
                "sealed_evaluation": True,
            },
        },
    }
    expected = {
        "game": game,
        "unit_type": GAME_CONTRACTS[game]["unit_type"],
        "mapped_fields": [],
        "disposition": "SEALED_EVALUATION",
        "evidence_authority": "ENVIRONMENT_VERIFIED",
        "should_parse": True,
        "should_abstain": True,
        "failure_modes": [],
    }
    return canary, expected


def build_corpus_projection(
    config_path: Path,
    aicourt_receipt_path: Path,
    aicourt_decisions_dir: Path,
    diplomacy_roots: Sequence[Path],
    diplomacy_manifest_paths: Sequence[Path],
    out_dir: Path,
    *,
    seed: int = 29,
    max_files_per_root: int = 64,
    max_records_per_file: int = 64,
    max_records_per_cohort: int = 24,
    max_record_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Create a bounded projection containing only allowlisted metadata and field-shape signals."""
    config = _load_config(config_path)
    receipt = load_structured(aicourt_receipt_path)
    allowed_cohorts = _allowed_cohorts(receipt)
    if len(diplomacy_roots) != len(diplomacy_manifest_paths):
        raise ValueError("each Diplomacy root requires one manifest")
    for root, manifest in zip(diplomacy_roots, diplomacy_manifest_paths, strict=True):
        if not root.is_dir() or not manifest.is_file() or not manifest.resolve().is_relative_to(root.resolve()):
            raise ValueError("Diplomacy manifest must exist inside its read-only corpus root")
        load_structured(manifest)
    if not aicourt_decisions_dir.is_dir():
        raise ValueError("AICourt decisions directory does not exist")
    if out_dir.exists():
        raise FileExistsError(f"projection output already exists: {out_dir}")

    reservoirs: dict[
        tuple[str, str, str],
        dict[str, tuple[str, dict[str, Any], dict[str, Any]]],
    ] = defaultdict(dict)
    source_files: list[tuple[Path, str | None]] = [
        (path, None)
        for path in sorted(aicourt_decisions_dir.glob("*.jsonl.gz"))[:max_files_per_root]
    ]
    for root in diplomacy_roots:
        source_files.extend(
            (path, "diplomacy")
            for path in sorted(root.rglob("*.jsonl.gz"))[:max_files_per_root]
        )
    for path, fallback_game in source_files:
        for _, raw, value in _iter_gzip_records(path, max_records_per_file, max_record_bytes):
            projected = _public_row(raw, value, config, fallback_game, allowed_cohorts)
            if projected is None:
                continue
            row, expected = projected
            agent = row["source_agent"]
            cohort = (row["game_hint"], agent["model_family"], agent["reasoning_effort"])
            item_id = str(row["item_id"])
            rank = sha256_json({"seed": seed, "source_locator_sha256": row["source_locator_sha256"]})
            bucket = reservoirs[cohort]
            bucket[item_id] = (rank, row, expected)
            if len(bucket) > max_records_per_cohort:
                worst_item = max(bucket, key=lambda key: bucket[key][0])
                del bucket[worst_item]
    selected = [
        (row, expected)
        for bucket in reservoirs.values()
        for _, row, expected in sorted(bucket.values(), key=lambda item: item[0])
    ]
    cohort_counts: Counter[tuple[str, str, str]] = Counter(
        {cohort: len(bucket) for cohort, bucket in reservoirs.items()}
    )
    present_games = {row["game_hint"] for row, _ in selected}
    if present_games != set(GAMES):
        raise ValueError(f"projection did not find all four games: {sorted(present_games)}")
    split_counts = Counter(row["split"] for row, _ in selected)
    for required_split, donor_split in (("ordinary_test", "ood_test"), ("ood_test", "ordinary_test")):
        if split_counts[required_split] == 0:
            donor = next((row for row, _ in reversed(selected) if row["split"] == donor_split), None)
            if donor is None or split_counts[donor_split] <= 1:
                raise ValueError("projection needs at least two real records for ordinary and OOD evaluation")
            donor["split"] = required_split
            split_counts[required_split] += 1
            split_counts[donor_split] -= 1

    canary_basis: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row, _ in selected:
        agent = row["source_agent"]
        key = (row["game_hint"], agent["model_family"], agent["reasoning_effort"])
        canary_basis.setdefault(key, row)
    selected.extend(_sealed_canary(row, seed) for row in canary_basis.values())
    selected.sort(key=lambda item: item[0]["item_id"])

    public_dir = out_dir / "public_dataset"
    steward_dir = out_dir / "private_steward"
    private_dir = out_dir / "private_projection"
    public_dir.mkdir(parents=True)
    steward_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    labels: list[dict[str, Any]] = []
    for public_input, expected in selected:
        record_hash = sha256_json(public_input)
        row = {**public_input, "record_hash": record_hash}
        rows_by_split[row["split"]].append(row)
        labels.append(
            {
                "schema_version": "game-adapter-ground-truth.v1",
                "item_id": row["item_id"],
                "record_hash": record_hash,
                "expected": expected,
                "authority": "ENVIRONMENT_OWNED",
                "verifier_id": config["verifier_id"],
            }
        )
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError("projection requires non-empty ordinary, OOD, and sealed splits")
    split_entries: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        path = public_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        split_entries[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(rows),
            "labels_in_file": False,
        }
    labels_path = steward_dir / "ground_truth.jsonl"
    _write_jsonl(labels_path, labels)
    manifest = {
        "schema_version": "game-adapter-dataset.v1",
        "dataset_id": f"content-free-corpus-projection-{seed}-{sha256_file(config_path)[:12]}",
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "content_policy": (
            "allowlisted cohort metadata, recognized field tokens, aggregate counts, typed flags, and hashes only; "
            "no raw corpus scalar values or filesystem paths"
        ),
        "games": list(GAMES),
        "splits": split_entries,
        "ground_truth_custody": "separate private steward file",
        "projection_authority": "aggregate receipts and manifests are claims; no record is hard-promoted",
    }
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    shutil.copyfile(config_path, out_dir / "frozen_verifier_config.yaml")
    dump_structured(
        steward_dir / "steward_receipt.json",
        {
            "schema_version": "1.0",
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "ground_truth_sha256": sha256_file(labels_path),
            "authority": "ENVIRONMENT_OWNED",
            "labels_exposed_to_candidates": False,
        },
    )
    dump_structured(
        private_dir / "projection_receipt.json",
        {
            "schema_version": "content-free-corpus-projection-receipt.v1",
            "aicourt_receipt_path": str(aicourt_receipt_path.resolve()),
            "aicourt_receipt_sha256": sha256_file(aicourt_receipt_path),
            "aicourt_decisions_dir": str(aicourt_decisions_dir.resolve()),
            "diplomacy_roots": [str(path.resolve()) for path in diplomacy_roots],
            "diplomacy_manifest_sha256": [sha256_file(path) for path in diplomacy_manifest_paths],
            "source_files_considered": len(source_files),
            "selected_real_records": sum(1 for row, _ in selected if row["projection_origin"] != "environment_canary"),
            "environment_canaries": sum(1 for row, _ in selected if row["projection_origin"] == "environment_canary"),
            "cohort_counts": {"|".join(key): value for key, value in sorted(cohort_counts.items())},
            "raw_values_emitted": False,
            "record_level_verification_claimed": False,
        },
    )
    return manifest


def _value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _path_allowed(game: str, field: str, path: tuple[str, ...]) -> bool:
    for pattern in SAFE_DERIVATION_PATHS[game].get(field, ()):
        if len(pattern) != len(path):
            continue
        matches = all(
            expected == "*" or expected == actual
            for expected, actual in zip(pattern, path, strict=True)
        )
        if matches:
            return True
    return False


def _subtree_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, nested in current.items():
                key_text = str(key)
                if SAFE_IDENTIFIER.fullmatch(key_text):
                    tokens.add(_token(key_text))
                stack.append(nested)
        elif isinstance(current, list):
            stack.extend(current[:128])
    return tokens


def _extract_safe_field_receipts(game: str, value: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    safe_fields = set(GAME_CONTRACTS[game]["safe_fields"])
    unsafe_tokens = HIDDEN_REASONING_TOKENS | PRIVATE_TOKENS
    receipts: dict[str, list[dict[str, str]]] = defaultdict(list)
    stack: list[tuple[tuple[str, ...], Any]] = [((), value)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, Mapping):
            for key, nested in current.items():
                key_text = str(key)
                normalized = _token(key_text) if SAFE_IDENTIFIER.fullmatch(key_text) else ""
                if normalized in unsafe_tokens:
                    continue
                child_path = (*path, normalized or "unrecognized")
                if (
                    normalized in safe_fields
                    and _path_allowed(game, normalized, child_path)
                    and not (_subtree_tokens(nested) & unsafe_tokens)
                ):
                    try:
                        value_sha256 = sha256_json(nested)
                    except ValueError:
                        continue
                    receipts[normalized].append(
                        {
                            "path_sha256": sha256_json(child_path),
                            "value_sha256": value_sha256,
                            "value_type": _value_kind(nested),
                        }
                    )
                    continue
                if isinstance(nested, (Mapping, list)):
                    stack.append((child_path, nested))
        elif isinstance(current, list):
            for index, nested in enumerate(current[:128]):
                if isinstance(nested, (Mapping, list)):
                    stack.append(((*path, f"item_{index}"), nested))
    return {
        field: sorted(items, key=lambda item: (item["path_sha256"], item["value_sha256"]))
        for field, items in sorted(receipts.items())
    }


def _safe_expected(game: str, safe_fields: Sequence[str]) -> dict[str, Any]:
    return {
        "game": game,
        "unit_type": GAME_CONTRACTS[game]["unit_type"],
        "mapped_fields": sorted(safe_fields),
        "disposition": "DISCOVERY_ONLY",
        "evidence_authority": "MODEL_INFERRED",
        "should_parse": True,
        "should_abstain": False,
        "failure_modes": [],
    }


def _safe_public_row(
    raw: bytes,
    value: Mapping[str, Any],
    fallback_game: str | None,
    allowed_cohorts: set[tuple[str, str]],
    source_path: Path,
    record_index: int,
    policy_version: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    game = _normalize_game(value.get("env"), fallback_game)
    if game is None:
        return None
    model = value.get("model") if value.get("model") in MODEL_FAMILIES else "unresolved"
    effort = value.get("reasoning_effort") if value.get("reasoning_effort") in REASONING_EFFORTS else "unknown"
    if game != "diplomacy" and (model, effort) not in allowed_cohorts:
        return None
    field_receipts = _extract_safe_field_receipts(game, value)
    if not field_receipts:
        return None
    source_content_sha256 = sha256_bytes(raw)
    lineage_group_sha256 = _lineage_group_sha256(value, source_path, game, policy_version)
    derivation_sha256 = sha256_json(
        {
            "policy_version": policy_version,
            "source_content_sha256": source_content_sha256,
            "lineage_group_sha256": lineage_group_sha256,
            "game": game,
            "safe_field_receipts": field_receipts,
        }
    )
    safe_fields = sorted(field_receipts)
    value_types = sorted({item["value_type"] for items in field_receipts.values() for item in items})
    source_agent = {
        "model_family": model,
        "reasoning_effort": effort,
        "metadata_authority": (
            "receipt_bound_manifest_claim" if game != "diplomacy" else "unresolved_manifest_claim"
        ),
    }
    public_input = {
        "schema_version": "game-adapter-input.v1",
        "item_id": f"safe-{derivation_sha256[:24]}",
        "split": "unassigned",
        "game_hint": game,
        "source_schema": f"{game}-safe-field-derivation-v1",
        "source_locator_sha256": derivation_sha256,
        "source_agent": source_agent,
        "projection_origin": "environment_safe_field_derivation",
        "source_shape": {
            "keys": safe_fields,
            "unknown_field_count": 0,
            "value_types": value_types,
            "record_kind": GAME_CONTRACTS[game]["unit_type"],
            "visibility": "public",
            "flags": {
                "aggregate_receipt_only": True,
                "environment_verified": False,
                "environment_verified_projection": True,
                "evaluation_material": False,
                "explicit_hidden_reasoning": False,
                "private_visibility": False,
                "safe_field_derivation": True,
                "schema_supported": True,
                "sealed_evaluation": False,
                "source_model_generated": True,
                "unresolved_fields": False,
            },
        },
    }
    ledger = {
        "schema_version": "safe-field-derivation-ledger.v1",
        "item_id": public_input["item_id"],
        "source_path": str(source_path.resolve()),
        "source_record_index": record_index,
        "source_content_sha256": source_content_sha256,
        "lineage_group_sha256": lineage_group_sha256,
        "derivation_sha256": derivation_sha256,
        "policy_version": policy_version,
        "game": game,
        "model_family": model,
        "reasoning_effort": effort,
        "safe_field_receipts": field_receipts,
        "raw_values_emitted": False,
    }
    return public_input, _safe_expected(game, safe_fields), ledger


def _lineage_group_sha256(
    value: Mapping[str, Any], source_path: Path, game: str, policy_version: str
) -> str:
    for field in ("episode_id", "game_id", "match_id"):
        identifier = value.get(field)
        if isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
            return sha256_json(
                {
                    "policy_version": policy_version,
                    "game": game,
                    "lineage_kind": field,
                    "identifier": identifier,
                }
            )
    return sha256_json(
        {
            "policy_version": policy_version,
            "game": game,
            "lineage_kind": "source_file",
            "source_path": str(source_path.resolve()),
        }
    )


def _assign_grouped_splits(
    selected: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], seed: int
) -> dict[str, Any]:
    parents = list(range(len(selected)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    by_lineage: dict[str, int] = {}
    by_content: dict[str, int] = {}
    for index, (_, _, ledger) in enumerate(selected):
        for value, lookup in (
            (str(ledger["lineage_group_sha256"]), by_lineage),
            (str(ledger["source_content_sha256"]), by_content),
        ):
            prior = lookup.setdefault(value, index)
            union(index, prior)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(selected)):
        components[find(index)].append(index)
    split_counts: Counter[str] = Counter()
    component_receipts: list[dict[str, Any]] = []
    for members in components.values():
        derivations = sorted(str(selected[index][2]["derivation_sha256"]) for index in members)
        split_group_sha256 = sha256_json(derivations)
        rank = sha256_json({"seed": seed, "split_group_sha256": split_group_sha256})
        split = "ordinary_test" if int(rank[:2], 16) < 192 else "ood_test"
        for index in members:
            row, _, ledger = selected[index]
            row["split"] = split
            ledger["split_group_sha256"] = split_group_sha256
            ledger["assigned_split"] = split
            split_counts[split] += 1
        component_receipts.append(
            {
                "split_group_sha256": split_group_sha256,
                "split": split,
                "records": len(members),
            }
        )
    if not {"ordinary_test", "ood_test"} <= set(split_counts):
        raise ValueError("lineage-grouped safe projection needs ordinary and OOD components")
    exact_duplicate_records = len(selected) - len(by_content)
    return {
        "lineage_groups": len(by_lineage),
        "split_components": len(components),
        "exact_duplicate_records": exact_duplicate_records,
        "split_counts": dict(sorted(split_counts.items())),
        "components": sorted(component_receipts, key=lambda item: item["split_group_sha256"]),
    }


def _read_indexed_gzip_record(path: Path, record_index: int, max_record_bytes: int) -> tuple[bytes, Any]:
    if record_index < 1:
        raise ValueError("source record index must be positive")
    with gzip.open(path, "rb") as handle:
        for index in range(1, record_index + 1):
            raw = handle.readline(max_record_bytes + 1)
            if not raw or len(raw) > max_record_bytes or not raw.endswith(b"\n"):
                raise ValueError(f"cannot re-read bounded source record {index}")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("safe derivation source record is not an object")
    return raw, value


def _verify_safe_derivations(
    ledger_path: Path,
    public_rows: Sequence[Mapping[str, Any]],
    approved_roots: Sequence[Path],
    max_record_bytes: int,
    out_path: Path,
) -> dict[str, Any]:
    row_by_id = {str(row["item_id"]): row for row in public_rows if row["projection_origin"] != "environment_canary"}
    verified = 0
    seen: set[str] = set()
    split_by_lineage: dict[str, str] = {}
    split_by_content: dict[str, str] = {}
    split_by_group: dict[str, str] = {}
    roots = [root.resolve() for root in approved_roots]
    for entry in _read_jsonl_private(ledger_path):
        item_id = str(entry.get("item_id", ""))
        if not item_id or item_id in seen or item_id not in row_by_id:
            raise ValueError(f"invalid safe derivation ledger item: {item_id}")
        seen.add(item_id)
        source_path = Path(str(entry["source_path"])).resolve()
        if not any(source_path.is_relative_to(root) for root in roots):
            raise ValueError(f"safe derivation source escapes approved roots: {item_id}")
        raw, value = _read_indexed_gzip_record(
            source_path, int(entry["source_record_index"]), max_record_bytes
        )
        if sha256_bytes(raw) != entry["source_content_sha256"]:
            raise ValueError(f"safe derivation source hash mismatch: {item_id}")
        game = str(entry["game"])
        receipts = _extract_safe_field_receipts(game, value)
        if receipts != entry["safe_field_receipts"]:
            raise ValueError(f"safe derivation field receipt mismatch: {item_id}")
        derivation_sha256 = sha256_json(
            {
                "policy_version": entry["policy_version"],
                "source_content_sha256": entry["source_content_sha256"],
                "lineage_group_sha256": entry["lineage_group_sha256"],
                "game": game,
                "safe_field_receipts": receipts,
            }
        )
        expected_lineage = _lineage_group_sha256(
            value, source_path, game, str(entry["policy_version"])
        )
        if expected_lineage != entry["lineage_group_sha256"]:
            raise ValueError(f"safe derivation lineage mismatch: {item_id}")
        row = row_by_id[item_id]
        _verify_public_safe_row(row, entry, receipts)
        assigned_split = str(entry["assigned_split"])
        for group, lookup in (
            (str(entry["lineage_group_sha256"]), split_by_lineage),
            (str(entry["source_content_sha256"]), split_by_content),
            (str(entry["split_group_sha256"]), split_by_group),
        ):
            prior_split = lookup.setdefault(group, assigned_split)
            if prior_split != assigned_split:
                raise ValueError(f"safe derivation split contamination: {item_id}")
        if derivation_sha256 != entry["derivation_sha256"] or derivation_sha256 != row["source_locator_sha256"]:
            raise ValueError(f"safe derivation binding mismatch: {item_id}")
        if sorted(receipts) != row["source_shape"]["keys"]:
            raise ValueError(f"safe derivation public field mismatch: {item_id}")
        verified += 1
    if seen != set(row_by_id):
        raise ValueError("safe derivation ledger does not cover all real public rows")
    receipt = {
        "schema_version": "safe-field-derivation-verification.v1",
        "authority": "ENVIRONMENT_OWNED",
        "verification_passed": True,
        "verified_records": verified,
        "verified_lineage_groups": len(split_by_lineage),
        "verified_split_groups": len(split_by_group),
        "split_contamination_detected": False,
        "ledger_sha256": sha256_file(ledger_path),
        "raw_values_emitted": False,
        "source_quality_verified": False,
        "claim_boundary": "Faithful safe-field structural extraction only; source claims remain model-inferred.",
    }
    dump_structured(out_path, receipt)
    return receipt


def _verify_public_safe_row(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    receipts: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    item_id = str(entry["item_id"])
    top_level_keys = {
        "game_hint",
        "item_id",
        "projection_origin",
        "record_hash",
        "schema_version",
        "source_agent",
        "source_locator_sha256",
        "source_schema",
        "source_shape",
        "split",
    }
    if set(row) != top_level_keys:
        raise ValueError(f"safe public row has unexpected fields: {item_id}")
    game = str(entry["game"])
    if (
        row["schema_version"] != "game-adapter-input.v1"
        or row["item_id"] != f"safe-{str(entry['derivation_sha256'])[:24]}"
        or row["game_hint"] != game
        or row["source_schema"] != f"{game}-safe-field-derivation-v1"
        or row["projection_origin"] != "environment_safe_field_derivation"
        or row["split"] != entry["assigned_split"]
    ):
        raise ValueError(f"safe public row contract mismatch: {item_id}")
    agent = row["source_agent"]
    expected_agent = {
        "model_family": entry["model_family"],
        "reasoning_effort": entry["reasoning_effort"],
        "metadata_authority": (
            "receipt_bound_manifest_claim" if game != "diplomacy" else "unresolved_manifest_claim"
        ),
    }
    if agent != expected_agent:
        raise ValueError(f"safe public row agent metadata mismatch: {item_id}")
    expected_flags = {
        "aggregate_receipt_only": True,
        "environment_verified": False,
        "environment_verified_projection": True,
        "evaluation_material": False,
        "explicit_hidden_reasoning": False,
        "private_visibility": False,
        "safe_field_derivation": True,
        "schema_supported": True,
        "sealed_evaluation": False,
        "source_model_generated": True,
        "unresolved_fields": False,
    }
    shape = row["source_shape"]
    expected_types = sorted({item["value_type"] for items in receipts.values() for item in items})
    if (
        set(shape)
        != {"flags", "keys", "record_kind", "unknown_field_count", "value_types", "visibility"}
        or shape["keys"] != sorted(receipts)
        or shape["unknown_field_count"] != 0
        or shape["value_types"] != expected_types
        or shape["record_kind"] != GAME_CONTRACTS[game]["unit_type"]
        or shape["visibility"] != "public"
        or shape["flags"] != expected_flags
        or set(shape["keys"]) - set(GAME_CONTRACTS[game]["safe_fields"])
    ):
        raise ValueError(f"safe public source shape mismatch: {item_id}")
    unhashed = dict(row)
    record_hash = str(unhashed.pop("record_hash", ""))
    if record_hash != sha256_json(unhashed):
        raise ValueError(f"safe public row hash mismatch: {item_id}")


def _read_jsonl_private(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected private JSON object at {path}:{line_number}")
            values.append(value)
    return values


def build_safe_field_projection(
    config_path: Path,
    aicourt_receipt_path: Path,
    aicourt_decisions_dir: Path,
    diplomacy_roots: Sequence[Path],
    diplomacy_manifest_paths: Sequence[Path],
    out_dir: Path,
    *,
    seed: int = 41,
    max_files_per_root: int = 64,
    max_records_per_file: int = 64,
    max_records_per_cohort: int = 16,
    max_record_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Build and environment-verify a content-bound, value-free safe-field adapter lane."""
    config = _load_config(config_path)
    receipt = load_structured(aicourt_receipt_path)
    allowed_cohorts = _allowed_cohorts(receipt)
    if len(diplomacy_roots) != len(diplomacy_manifest_paths):
        raise ValueError("each Diplomacy root requires one manifest")
    for root, manifest in zip(diplomacy_roots, diplomacy_manifest_paths, strict=True):
        if not root.is_dir() or not manifest.is_file() or not manifest.resolve().is_relative_to(root.resolve()):
            raise ValueError("Diplomacy manifest must exist inside its read-only corpus root")
        load_structured(manifest)
    if not aicourt_decisions_dir.is_dir():
        raise ValueError("AICourt decisions directory does not exist")
    if out_dir.exists():
        raise FileExistsError(f"projection output already exists: {out_dir}")

    policy_sha256 = sha256_json(
        {
            "config_sha256": sha256_file(config_path),
            "safe_fields": {game: sorted(GAME_CONTRACTS[game]["safe_fields"]) for game in GAMES},
            "hidden_reasoning_tokens": sorted(HIDDEN_REASONING_TOKENS),
            "private_tokens": sorted(PRIVATE_TOKENS),
            "safe_derivation_paths": SAFE_DERIVATION_PATHS,
        }
    )
    policy_version = f"safe-field-derivation.v1:{policy_sha256}"
    reservoirs: dict[
        tuple[str, str, str],
        dict[str, tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]],
    ] = defaultdict(dict)
    source_files: list[tuple[Path, str | None]] = [
        (path, None)
        for path in sorted(aicourt_decisions_dir.glob("*.jsonl.gz"))[:max_files_per_root]
    ]
    for root in diplomacy_roots:
        source_files.extend(
            (path, "diplomacy") for path in sorted(root.rglob("*.jsonl.gz"))[:max_files_per_root]
        )
    for path, fallback_game in source_files:
        for record_index, raw, value in _iter_gzip_records(path, max_records_per_file, max_record_bytes):
            projected = _safe_public_row(
                raw,
                value,
                fallback_game,
                allowed_cohorts,
                path,
                record_index,
                policy_version,
            )
            if projected is None:
                continue
            row, expected, ledger = projected
            agent = row["source_agent"]
            cohort = (row["game_hint"], agent["model_family"], agent["reasoning_effort"])
            item_id = str(row["item_id"])
            rank = sha256_json({"seed": seed, "derivation_sha256": row["source_locator_sha256"]})
            bucket = reservoirs[cohort]
            bucket[item_id] = (rank, row, expected, ledger)
            if len(bucket) > max_records_per_cohort:
                worst_item = max(bucket, key=lambda key: bucket[key][0])
                del bucket[worst_item]
    selected = [
        (row, expected, ledger)
        for bucket in reservoirs.values()
        for _, row, expected, ledger in sorted(bucket.values(), key=lambda item: item[0])
    ]
    present_games = {row["game_hint"] for row, _, _ in selected}
    if present_games != set(GAMES):
        raise ValueError(f"safe projection did not find all four games: {sorted(present_games)}")
    split_group_receipt = _assign_grouped_splits(selected, seed)

    canary_basis: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row, _, _ in selected:
        agent = row["source_agent"]
        key = (row["game_hint"], agent["model_family"], agent["reasoning_effort"])
        canary_basis.setdefault(key, row)
    projected_rows: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (row, expected) for row, expected, _ in selected
    ]
    projected_rows.extend(_sealed_canary(row, seed) for row in canary_basis.values())
    projected_rows.sort(key=lambda item: item[0]["item_id"])

    public_dir = out_dir / "public_dataset"
    steward_dir = out_dir / "private_steward"
    private_dir = out_dir / "private_projection"
    public_dir.mkdir(parents=True)
    steward_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    labels: list[dict[str, Any]] = []
    all_public_rows: list[dict[str, Any]] = []
    for public_input, expected in projected_rows:
        record_hash = sha256_json(public_input)
        row = {**public_input, "record_hash": record_hash}
        rows_by_split[row["split"]].append(row)
        all_public_rows.append(row)
        labels.append(
            {
                "schema_version": "game-adapter-ground-truth.v1",
                "item_id": row["item_id"],
                "record_hash": record_hash,
                "expected": expected,
                "authority": "ENVIRONMENT_OWNED",
                "verifier_id": config["verifier_id"],
            }
        )
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError("safe projection requires non-empty ordinary, OOD, and sealed splits")
    split_entries: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        path = public_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        split_entries[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(rows),
            "labels_in_file": False,
        }
    labels_path = steward_dir / "ground_truth.jsonl"
    _write_jsonl(labels_path, labels)
    ledger_path = private_dir / "derivation_ledger.jsonl"
    _write_jsonl(ledger_path, [ledger for _, _, ledger in sorted(selected, key=lambda item: item[0]["item_id"])])
    split_group_path = private_dir / "split_group_receipt.json"
    dump_structured(
        split_group_path,
        {
            "schema_version": "safe-field-split-groups.v1",
            **split_group_receipt,
            "split_contamination_detected": False,
        },
    )
    verification_path = steward_dir / "derivation_verification.json"
    _verify_safe_derivations(
        ledger_path,
        all_public_rows,
        [aicourt_decisions_dir, *diplomacy_roots],
        max_record_bytes,
        verification_path,
    )
    manifest = {
        "schema_version": "game-adapter-dataset.v1",
        "dataset_id": f"safe-field-corpus-projection-{seed}-{sha256_file(config_path)[:12]}",
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "content_policy": (
            "content-bound allowlisted safe-field names, types, cohort metadata, flags, and derivation hashes only; "
            "no raw corpus scalar values or filesystem paths"
        ),
        "games": list(GAMES),
        "splits": split_entries,
        "ground_truth_custody": "separate private steward file",
        "projection_authority": (
            "environment-owned derivation fidelity verified; source gameplay claims remain model-inferred"
        ),
        "projection_mode": "safe_field_derivation",
        "split_grouping": "episode/game lineage plus exact-content connected components",
        "split_group_receipt_sha256": sha256_file(split_group_path),
        "derivation_verification_sha256": sha256_file(verification_path),
    }
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    shutil.copyfile(config_path, out_dir / "frozen_verifier_config.yaml")
    dump_structured(
        steward_dir / "steward_receipt.json",
        {
            "schema_version": "1.0",
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "ground_truth_sha256": sha256_file(labels_path),
            "derivation_verification_sha256": sha256_file(verification_path),
            "authority": "ENVIRONMENT_OWNED",
            "labels_exposed_to_candidates": False,
        },
    )
    cohort_counts = Counter(
        (
            row["game_hint"],
            row["source_agent"]["model_family"],
            row["source_agent"]["reasoning_effort"],
        )
        for row, _, _ in selected
    )
    dump_structured(
        private_dir / "projection_receipt.json",
        {
            "schema_version": "safe-field-corpus-projection-receipt.v1",
            "aicourt_receipt_path": str(aicourt_receipt_path.resolve()),
            "aicourt_receipt_sha256": sha256_file(aicourt_receipt_path),
            "aicourt_decisions_dir": str(aicourt_decisions_dir.resolve()),
            "diplomacy_roots": [str(path.resolve()) for path in diplomacy_roots],
            "diplomacy_manifest_sha256": [sha256_file(path) for path in diplomacy_manifest_paths],
            "source_files_considered": len(source_files),
            "selected_real_records": len(selected),
            "environment_canaries": len(canary_basis),
            "cohort_counts": {"|".join(key): value for key, value in sorted(cohort_counts.items())},
            "derivation_ledger_sha256": sha256_file(ledger_path),
            "split_group_receipt_sha256": sha256_file(split_group_path),
            "lineage_groups": split_group_receipt["lineage_groups"],
            "split_components": split_group_receipt["split_components"],
            "exact_duplicate_records": split_group_receipt["exact_duplicate_records"],
            "split_contamination_detected": False,
            "derivation_verification_sha256": sha256_file(verification_path),
            "raw_values_emitted": False,
            "record_level_derivation_verified": True,
            "source_quality_verified": False,
        },
    )
    return manifest
