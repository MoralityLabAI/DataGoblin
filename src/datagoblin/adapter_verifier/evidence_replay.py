from __future__ import annotations

import gzip
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.data_etiquette.evidence_receipts import (
    EvidenceVerificationReceipt,
    EvidenceVerifierRegistry,
    issue_evidence_receipt,
)
from datagoblin.domain.evidence import EvidenceType
from datagoblin.hashing import sha256_bytes, sha256_file, sha256_json
from datagoblin.serialization import dump_structured, to_jsonable

from .corpus_projection import _extract_safe_field_receipts, _read_indexed_gzip_record

GAME_EVIDENCE = {
    "centauri": (
        "centauri-environment-replay-v1",
        "centauri-state-transition-replay-v1",
        EvidenceType.ENVIRONMENT_VERIFIED,
    ),
    "they_sing": (
        "they-sing-environment-replay-v1",
        "they-sing-state-transition-replay-v1",
        EvidenceType.ENVIRONMENT_VERIFIED,
    ),
    "court": (
        "court-environment-replay-v1",
        "court-state-transition-replay-v1",
        EvidenceType.ENVIRONMENT_VERIFIED,
    ),
    "diplomacy": (
        "diplomacy-adjudication-replay-v1",
        "diplomacy-adjudication-replay-v1",
        EvidenceType.INDEPENDENT_HARNESS,
    ),
}
POWERS = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")
CLAIM_TYPES = ("RULE_COMPLIANCE", "OUTCOME_VALIDITY", "STATE_GROUNDING")


@dataclass(frozen=True)
class ReplayResult:
    verdict: str
    reason: str
    checks: Mapping[str, bool]
    artifact_sha256s: tuple[str, ...]
    details: Mapping[str, Any]


def verify_semantic_evidence(
    challenges_path: Path,
    review_map_path: Path,
    derivation_ledger_path: Path,
    evidence_registry_path: Path,
    approved_roots: Sequence[Path],
    out_dir: Path,
    *,
    aicourt_corpus_root: Path,
    centauri_root: Path,
    they_sing_root: Path,
    diplomacy_engine_root: Path,
    diplomacy_corpus_roots: Sequence[Path],
    court_engine_snapshots: Mapping[str, Path],
    method_specs_dir: Path | None = None,
    max_record_bytes: int = 2_000_000,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Replay a semantic-review challenge set without publishing private values or paths."""
    if out_dir.exists():
        raise FileExistsError(f"evidence replay result already exists: {out_dir}")
    registry = EvidenceVerifierRegistry(evidence_registry_path)
    specs_dir = method_specs_dir or evidence_registry_path.parent / "evidence_methods"
    _validate_method_specs(registry, specs_dir)

    challenges = _read_jsonl(challenges_path)
    review_rows = _unique_review_rows(_read_jsonl(review_map_path))
    ledger_rows = _unique_by_item(_read_jsonl(derivation_ledger_path), "derivation ledger")
    roots = tuple(path.resolve() for path in approved_roots)
    diplomacy_roots = tuple(path.resolve() for path in diplomacy_corpus_roots)
    if not roots or not diplomacy_roots:
        raise ValueError("approved roots and Diplomacy corpus roots are required")

    out_dir.mkdir(parents=True)
    observation_dir = out_dir / "private_observations"
    input_dir = out_dir / "private_replay_inputs"
    report_dir = out_dir / "private_environment_reports"
    observation_dir.mkdir()
    input_dir.mkdir()
    report_dir.mkdir()

    cache: dict[tuple[str, str], ReplayResult] = {}
    receipts: list[EvidenceVerificationReceipt] = []
    counts: Counter[tuple[str, str]] = Counter()
    for challenge in sorted(challenges, key=lambda row: (str(row["game"]), str(row["item_id"]))):
        item_id = str(challenge["item_id"])
        map_row = review_rows.get(item_id)
        ledger = ledger_rows.get(item_id)
        if map_row is None or ledger is None:
            raise ValueError(f"challenge item is absent from private bindings: {item_id}")
        game = str(challenge["game"])
        if game not in GAME_EVIDENCE:
            raise ValueError(f"unsupported evidence game: {game}")
        _validate_challenge_binding(challenge, map_row, ledger)
        source_path = Path(str(ledger["source_path"])).resolve()
        if not _is_beneath(source_path, roots):
            raise ValueError(f"challenge source escapes approved roots: {item_id}")
        raw, record = _read_indexed_gzip_record(
            source_path,
            int(ledger["source_record_index"]),
            max_record_bytes,
        )
        if sha256_bytes(raw) != str(ledger["source_content_sha256"]):
            raise ValueError(f"challenge source content mismatch: {item_id}")
        if _extract_safe_field_receipts(game, record) != ledger["safe_field_receipts"]:
            raise ValueError(f"challenge safe-field commitment mismatch: {item_id}")

        cache_key = _replay_cache_key(game, source_path, record)
        result = cache.get(cache_key)
        if result is None:
            result = _run_game_replay(
                game,
                source_path,
                record,
                aicourt_corpus_root=aicourt_corpus_root.resolve(),
                centauri_root=centauri_root.resolve(),
                they_sing_root=they_sing_root.resolve(),
                diplomacy_engine_root=diplomacy_engine_root.resolve(),
                diplomacy_corpus_roots=diplomacy_roots,
                court_engine_snapshots={
                    str(version): path.resolve() for version, path in court_engine_snapshots.items()
                },
                input_dir=input_dir,
                report_dir=report_dir,
                timeout_seconds=timeout_seconds,
            )
            cache[cache_key] = result

        artifact_sha256s = tuple(
            sorted({str(ledger["source_content_sha256"]), *result.artifact_sha256s})
        )
        observation = {
            "schema_version": "game-evidence-observation.v1",
            "item_id": item_id,
            "subject_sha256": str(challenge["subject_sha256"]),
            "game": game,
            "verdict": result.verdict,
            "reason": result.reason,
            "checks": dict(sorted(result.checks.items())),
            "details": to_jsonable(result.details),
            "source_record_sha256": str(ledger["source_content_sha256"]),
            "artifact_sha256s": list(artifact_sha256s),
            "raw_values_emitted": False,
            "source_paths_emitted": False,
        }
        observation_path = observation_dir / f"{item_id}.json"
        dump_structured(observation_path, observation)

        verifier_id, method_id, evidence_type = GAME_EVIDENCE[game]
        receipt = issue_evidence_receipt(
            registry,
            subject_sha256=str(challenge["subject_sha256"]),
            evidence_type=evidence_type,
            verifier_id=verifier_id,
            method_id=method_id,
            asserted_by=str(challenge["asserted_by"]),
            verdict=result.verdict,
            challenge_sha256=sha256_json(challenge),
            observation_sha256=sha256_file(observation_path),
            claim_types=CLAIM_TYPES,
            artifact_sha256s=artifact_sha256s,
        )
        receipts.append(receipt)
        counts[(game, result.verdict)] += 1

    receipts_path = out_dir / "evidence_receipts.jsonl"
    _write_jsonl(receipts_path, [receipt.to_jsonable() for receipt in receipts])
    passing = sum(receipt.verdict == "PASS" for receipt in receipts)
    report = {
        "schema_version": "game-evidence-replay-report.v1",
        "challenge_manifest_sha256": sha256_file(challenges_path),
        "review_map_sha256": sha256_file(review_map_path),
        "derivation_ledger_sha256": sha256_file(derivation_ledger_path),
        "registry_sha256": registry.sha256,
        "method_spec_sha256s": {
            method_id: sha256_file(specs_dir / f"{method_id}.yaml")
            for _, method_id, _ in GAME_EVIDENCE.values()
        },
        "challenges": len(challenges),
        "passing_receipts": passing,
        "inconclusive_receipts": len(receipts) - passing,
        "coverage": passing / len(receipts) if receipts else 0.0,
        "by_game": {
            game: {
                "challenges": sum(value for (name, _), value in counts.items() if name == game),
                "passing": counts[(game, "PASS")],
                "inconclusive": counts[(game, "INCONCLUSIVE")],
            }
            for game in sorted(GAME_EVIDENCE)
        },
        "evidence_receipts_sha256": sha256_file(receipts_path),
        "private_observation_count": len(receipts),
        "raw_values_emitted": False,
        "source_paths_emitted": False,
        "training_release_authorized": False,
        "non_claims": [
            "Passing receipts cover only the bounded semantic-review sample.",
            "Inconclusive replay under a drifted engine is not evidence that the original outcome was false.",
            "The local registry and receipt store require independent operational custody for production use.",
        ],
    }
    dump_structured(out_dir / "evidence_replay_report.json", report)
    return report


def _run_game_replay(
    game: str,
    source_path: Path,
    record: Mapping[str, Any],
    **kwargs: Any,
) -> ReplayResult:
    if game == "centauri":
        return _replay_centauri(source_path, record, **kwargs)
    if game == "they_sing":
        return _replay_they_sing(source_path, record, **kwargs)
    if game == "court":
        return _replay_court(source_path, record, **kwargs)
    return _replay_diplomacy(source_path, record, **kwargs)


def _replay_centauri(
    source_path: Path,
    record: Mapping[str, Any],
    *,
    aicourt_corpus_root: Path,
    centauri_root: Path,
    report_dir: Path,
    timeout_seconds: int,
    **_: Any,
) -> ReplayResult:
    native_root = aicourt_corpus_root / "native"
    native_dir = _native_dir(native_root, source_path, "centauri")
    dataset = _find_centauri_dataset(
        native_root,
        native_dir / "native-centauri.jsonl",
        str(record.get("episode_id", "")),
    )
    if dataset is None:
        return _inconclusive("NATIVE_EPISODE_MISSING", ())
    manifest = native_dir / "native-centauri.manifest.json"
    if dataset.parent != native_dir:
        manifest = dataset.with_name("native-centauri.manifest.json")
    harness = centauri_root / "tools" / "harness.mjs"
    artifacts = _existing_hashes(
        dataset,
        manifest,
        harness,
        centauri_root / "tools" / "research_dataset.mjs",
        centauri_root / "src" / "research_v2.mjs",
        centauri_root / "src" / "engine.mjs",
    )
    if not dataset.is_file() or not harness.is_file() or shutil.which("node") is None:
        return _inconclusive("ENVIRONMENT_OR_NATIVE_ARTIFACT_MISSING", artifacts)

    coordinates = _bind_centauri_record(dataset, record)
    if not coordinates:
        return ReplayResult(
            "INCONCLUSIVE",
            "NATIVE_DECISION_BINDING_MISMATCH",
            {"native_decision_bound": False},
            artifacts,
            {},
        )
    completed = _run_process(
        [
            "node",
            str(harness),
            "research-replay",
            "--dataset",
            str(dataset),
            "--episode",
            str(record.get("episode_id", "")),
        ],
        cwd=centauri_root,
        timeout_seconds=timeout_seconds,
    )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raw_report = {}
    sanitized = {
        "schema_version": "centauri-environment-replay-observation.v1",
        "return_code": completed.returncode,
        "native_decision_bound": coordinates,
        "replay_ok": raw_report.get("ok") is True,
        "turns_replayed": int(raw_report.get("turns_replayed", 0) or 0),
        "mismatch_count": (
            len(raw_report.get("mismatches", []))
            if isinstance(raw_report.get("mismatches"), list)
            else None
        ),
        "runner_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
    }
    report_path = report_dir / f"centauri-{sha256_json(sanitized)[:24]}.json"
    dump_structured(report_path, sanitized)
    artifacts = tuple(sorted({*artifacts, sha256_file(report_path)}))
    passed = completed.returncode == 0 and sanitized["replay_ok"]
    return ReplayResult(
        "PASS" if passed else "INCONCLUSIVE",
        "REPLAY_MATCH" if passed else "ENGINE_REPLAY_MISMATCH",
        {
            "native_decision_bound": coordinates,
            "environment_replay_passed": passed,
        },
        artifacts,
        {
            "turns_replayed": sanitized["turns_replayed"],
            "mismatch_count": sanitized["mismatch_count"],
        },
    )


def _replay_they_sing(
    source_path: Path,
    record: Mapping[str, Any],
    *,
    aicourt_corpus_root: Path,
    they_sing_root: Path,
    input_dir: Path,
    report_dir: Path,
    timeout_seconds: int,
    **_: Any,
) -> ReplayResult:
    native_root = aicourt_corpus_root / "native"
    native_dir = _native_dir(native_root, source_path, "they-sing")
    native_log = _find_they_sing_log(
        native_root,
        native_dir,
        str(record.get("episode_id", "")),
    )
    replay_script = they_sing_root / "scripts" / "replay-harness-run.cjs"
    session_module = they_sing_root / "dist-harness" / "harness" / "HeadlessPlaytestSession.js"
    artifacts = _existing_hashes(replay_script, session_module)
    if native_log is None or not replay_script.is_file() or shutil.which("node") is None:
        return _inconclusive("ENVIRONMENT_OR_NATIVE_ARTIFACT_MISSING", artifacts)
    native_raw = native_log.read_bytes()
    artifacts = tuple(sorted({*artifacts, sha256_bytes(native_raw)}))
    native_rows = [
        json.loads(line)
        for line in native_raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    source_bound = _bind_they_sing_record(native_rows, record)
    if not source_bound:
        return ReplayResult(
            "INCONCLUSIVE",
            "NATIVE_DECISION_BINDING_MISMATCH",
            {"native_decision_bound": False},
            artifacts,
            {},
        )
    created = next((row for row in native_rows if row.get("type") == "session_created"), None)
    if not isinstance(created, Mapping) or not isinstance(created.get("data"), Mapping):
        return _inconclusive("SESSION_CONFIGURATION_MISSING", artifacts)
    data = created["data"]
    config = {
        key: data[key]
        for key in ("name", "maxTurns", "seed", "enforcementMode", "factionLabels", "agents")
        if key in data
    }
    config_path = input_dir / f"they-sing-{sha256_json(config)[:24]}.json"
    if not config_path.exists():
        dump_structured(config_path, config)
    external_report_path = report_dir / f"they-sing-raw-{sha256_bytes(native_raw)[:24]}.json"
    completed = _run_process(
        [
            "node",
            str(replay_script),
            "--run",
            str(native_log),
            "--config",
            str(config_path),
            "--output",
            str(external_report_path),
        ],
        cwd=they_sing_root,
        timeout_seconds=timeout_seconds,
    )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raw_report = {}
    replay_log = Path(str(raw_report.get("replayLogFile", "")))
    replay_turn_commitment = _they_sing_turn_commitment(replay_log)
    sanitized = {
        "schema_version": "they-sing-environment-replay-observation.v1",
        "return_code": completed.returncode,
        "native_decision_bound": source_bound,
        "status": str(raw_report.get("status", "")),
        "turns_compared": int(raw_report.get("turnsCompared", 0) or 0),
        "mismatch_count": (
            len(raw_report.get("mismatches", []))
            if isinstance(raw_report.get("mismatches"), list)
            else None
        ),
        "replay_turn_commitment_sha256": replay_turn_commitment,
        "runner_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
    }
    sanitized_path = report_dir / f"they-sing-{sha256_json(sanitized)[:24]}.json"
    dump_structured(sanitized_path, sanitized)
    replay_hashes = (replay_turn_commitment,) if replay_turn_commitment is not None else ()
    artifacts = tuple(
        sorted(
            {
                *artifacts,
                sha256_file(config_path),
                sha256_file(sanitized_path),
                *replay_hashes,
            }
        )
    )
    passed = (
        completed.returncode == 0
        and raw_report.get("status") == "passed"
        and sanitized["turns_compared"] > 0
        and sanitized["mismatch_count"] == 0
        and replay_turn_commitment is not None
    )
    return ReplayResult(
        "PASS" if passed else "INCONCLUSIVE",
        "REPLAY_MATCH" if passed else "ENGINE_REPLAY_MISMATCH",
        {
            "native_decision_bound": source_bound,
            "environment_replay_passed": passed,
        },
        artifacts,
        {
            "turns_compared": sanitized["turns_compared"],
            "mismatch_count": sanitized["mismatch_count"],
        },
    )


def _replay_court(
    source_path: Path,
    record: Mapping[str, Any],
    *,
    aicourt_corpus_root: Path,
    court_engine_snapshots: Mapping[str, Path],
    report_dir: Path,
    timeout_seconds: int,
    **_: Any,
) -> ReplayResult:
    stem = _gzip_jsonl_stem(source_path)
    canonical = aicourt_corpus_root / "canonical" / f"{stem}.jsonl.gz"
    artifacts = _existing_hashes(source_path, canonical)
    if not canonical.is_file():
        return _inconclusive("CANONICAL_EPISODE_MISSING", artifacts)
    canonical_rows = _read_gzip_rows(canonical)
    if len(canonical_rows) != 1 or canonical_rows[0].get("episode_id") != record.get("episode_id"):
        return ReplayResult(
            "INCONCLUSIVE",
            "CANONICAL_EPISODE_BINDING_MISMATCH",
            {"canonical_episode_bound": False},
            artifacts,
            {},
        )
    version = str(canonical_rows[0].get("env_version", ""))
    engine_path = court_engine_snapshots.get(version)
    if engine_path is None or not engine_path.is_file() or shutil.which("node") is None:
        return ReplayResult(
            "INCONCLUSIVE",
            "EXACT_ENGINE_SNAPSHOT_MISSING",
            {"canonical_episode_bound": True, "exact_engine_available": False},
            artifacts,
            {"declared_engine_version": version},
        )
    artifacts = tuple(sorted({*artifacts, sha256_file(engine_path)}))
    try:
        episode_index = int(stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return _inconclusive("EPISODE_INDEX_UNAVAILABLE", artifacts)
    helper = Path(__file__).with_name("court_replay.mjs")
    completed = _run_process(
        [
            "node",
            str(helper),
            "--engine",
            str(engine_path),
            "--canonical",
            str(canonical),
            "--decisions",
            str(source_path),
            "--episode-index",
            str(episode_index),
        ],
        cwd=engine_path.parent.parent.parent,
        timeout_seconds=timeout_seconds,
    )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raw_report = {}
    sanitized = {
        key: value
        for key, value in raw_report.items()
        if key
        in {
            "schema_version",
            "status",
            "reason",
            "declared_engine_version",
            "available_engine_version",
            "decisions",
            "decisions_consumed",
            "checks",
            "canonical_sha256",
            "decisions_sha256",
            "engine_sha256",
        }
    }
    sanitized["return_code"] = completed.returncode
    sanitized["runner_output_sha256"] = sha256_bytes(completed.stdout.encode("utf-8"))
    sanitized["runner_stderr_sha256"] = sha256_bytes(completed.stderr.encode("utf-8"))
    report_path = report_dir / f"court-{sha256_json(sanitized)[:24]}.json"
    dump_structured(report_path, sanitized)
    artifacts = tuple(sorted({*artifacts, sha256_file(report_path)}))
    passed = completed.returncode == 0 and raw_report.get("status") == "PASS"
    checks = raw_report.get("checks", {})
    safe_checks = (
        {str(key): value is True for key, value in checks.items()}
        if isinstance(checks, Mapping)
        else {}
    )
    safe_checks.update(
        {
            "canonical_episode_bound": True,
            "exact_engine_available": True,
            "environment_replay_passed": passed,
        }
    )
    return ReplayResult(
        "PASS" if passed else "INCONCLUSIVE",
        "REPLAY_MATCH" if passed else str(raw_report.get("reason", "ENVIRONMENT_RUNNER_ERROR")),
        safe_checks,
        artifacts,
        {
            "declared_engine_version": version,
            "decisions": int(raw_report.get("decisions", 0) or 0),
            "decisions_consumed": int(raw_report.get("decisions_consumed", 0) or 0),
        },
    )


def _replay_diplomacy(
    source_path: Path,
    record: Mapping[str, Any],
    *,
    diplomacy_engine_root: Path,
    diplomacy_corpus_roots: Sequence[Path],
    report_dir: Path,
    **_: Any,
) -> ReplayResult:
    corpus_root = next(
        (root for root in diplomacy_corpus_roots if source_path.is_relative_to(root)),
        None,
    )
    if corpus_root is None:
        return _inconclusive("DIPLOMACY_CORPUS_ROOT_MISMATCH", (sha256_file(source_path),))
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        return _inconclusive("PROVENANCE_BINDING_MISSING", (sha256_file(source_path),))
    archive_value = Path(str(provenance.get("archive", "")))
    if archive_value.is_absolute():
        return _inconclusive("ABSOLUTE_PROVENANCE_PATH_REJECTED", (sha256_file(source_path),))
    archive = (corpus_root / archive_value).resolve()
    if not archive.is_relative_to(corpus_root) or not archive.is_file():
        return _inconclusive("PROVENANCE_ARCHIVE_MISSING", (sha256_file(source_path),))
    archive_sha256 = sha256_file(archive)
    artifacts = (archive_sha256, sha256_file(source_path))
    archive_bound = archive_sha256 == str(provenance.get("sha256", ""))
    if not archive_bound:
        return ReplayResult(
            "INCONCLUSIVE",
            "PROVENANCE_ARCHIVE_HASH_MISMATCH",
            {"provenance_archive_bound": False},
            artifacts,
            {},
        )
    try:
        with tarfile.open(archive, "r:gz") as handle:
            member = handle.extractfile("lmvsgame.json")
            if member is None:
                raise KeyError("lmvsgame.json")
            saved_raw = member.read()
        saved_game = json.loads(saved_raw)
        is_valid_saved_game = _load_diplomacy_validator(diplomacy_engine_root)
        replayed = bool(is_valid_saved_game(saved_game))
        projected = _project_diplomacy_outcomes(saved_game)
        projection_bound = projected == record.get("trace", {}).get("adjudicated_outcomes")
    except (ImportError, KeyError, OSError, tarfile.TarError, json.JSONDecodeError, ValueError):
        return _inconclusive("ADJUDICATOR_REPLAY_ERROR", artifacts)
    saved_sha256 = sha256_bytes(saved_raw)
    adjudicator_hashes = _existing_hashes(
        diplomacy_engine_root / "diplomacy" / "utils" / "export.py",
        diplomacy_engine_root / "diplomacy" / "engine" / "game.py",
        diplomacy_engine_root / "diplomacy" / "engine" / "map.py",
    )
    sanitized = {
        "schema_version": "diplomacy-adjudication-replay-observation.v1",
        "provenance_archive_bound": archive_bound,
        "saved_game_replayed": replayed,
        "public_projection_bound": projection_bound,
        "adjudicated_outcomes": len(projected),
        "saved_game_sha256": saved_sha256,
    }
    report_path = report_dir / f"diplomacy-{sha256_json(sanitized)[:24]}.json"
    dump_structured(report_path, sanitized)
    artifacts = tuple(
        sorted({*artifacts, *adjudicator_hashes, saved_sha256, sha256_file(report_path)})
    )
    passed = archive_bound and replayed and projection_bound
    return ReplayResult(
        "PASS" if passed else "INCONCLUSIVE",
        "REPLAY_MATCH" if passed else "ADJUDICATION_OR_PROJECTION_MISMATCH",
        {
            "provenance_archive_bound": archive_bound,
            "saved_game_replayed": replayed,
            "public_projection_bound": projection_bound,
        },
        artifacts,
        {
            "adjudicated_outcomes": len(projected),
        },
    )


def _validate_method_specs(registry: EvidenceVerifierRegistry, specs_dir: Path) -> None:
    for verifier in registry.verifiers.values():
        for method_id, method in verifier["methods"].items():
            spec_path = specs_dir / f"{method_id}.yaml"
            if not spec_path.is_file():
                raise ValueError(f"evidence replay method specification missing: {method_id}")
            if sha256_file(spec_path) != method["replay_spec_sha256"]:
                raise ValueError(f"evidence replay method specification hash mismatch: {method_id}")


def _validate_challenge_binding(
    challenge: Mapping[str, Any],
    map_row: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> None:
    game = str(challenge["game"])
    verifier_id, method_id, _ = GAME_EVIDENCE[game]
    checks = {
        "subject": challenge.get("subject_sha256") == map_row.get("record_hash"),
        "asserted_by": challenge.get("asserted_by") == map_row.get("model_family"),
        "game_map": game == map_row.get("game"),
        "game_ledger": game == ledger.get("game"),
        "item_map": challenge.get("item_id") == map_row.get("item_id"),
        "item_ledger": challenge.get("item_id") == ledger.get("item_id"),
        "verifier": challenge.get("required_verifier_id") == verifier_id,
        "method": challenge.get("required_method_id") == method_id,
        "claims": challenge.get("required_claim_types") == ["OUTCOME_VALIDITY"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"evidence challenge binding mismatch for {challenge.get('item_id')}: {','.join(failed)}"
        )


def _bind_centauri_record(dataset: Path, record: Mapping[str, Any]) -> bool:
    matches = 0
    with dataset.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_id") != record.get("episode_id") or row.get("event") != "step":
                continue
            payload = row.get("payload", {})
            if payload.get("turn") != record.get("turn"):
                continue
            for phase in payload.get("phases", []):
                if phase.get("phase") != record.get("phase"):
                    continue
                for decision in phase.get("decisions", []):
                    request = decision.get("request", {})
                    if (
                        decision.get("actor_id") == record.get("seat")
                        and request.get("legal_commands") == record.get("legal_actions")
                    ):
                        matches += 1
    return matches == 1


def _bind_they_sing_record(
    native_rows: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
) -> bool:
    created = [row for row in native_rows if row.get("type") == "session_created"]
    if len(created) != 1 or created[0].get("sessionId") != record.get("episode_id"):
        return False
    phase_orders: dict[str, Any] = {}
    expected_phases = {
        "ALLOCATION": "allocation",
        "ACTION_DECLARATION": "action",
    }
    for row in native_rows:
        if (
            row.get("type") != "phase_reasoning_diary"
            or row.get("turn") != record.get("turn")
            or not isinstance(row.get("data"), Mapping)
            or row["data"].get("factionId") != record.get("seat")
        ):
            continue
        target_key = expected_phases.get(str(row.get("phase")))
        if target_key:
            phase_orders[target_key] = row["data"].get("requestedOrders")
    target = record.get("target")
    if not isinstance(target, Mapping):
        return False
    return all(
        isinstance(target.get(key), Mapping)
        and target[key].get("orders") == phase_orders.get(key)
        for key in ("allocation", "action")
    )


def _project_diplomacy_outcomes(saved_game: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    phases = saved_game.get("phases", [])
    if not isinstance(phases, list):
        return outcomes
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        phase_name = str(phase.get("name", ""))
        orders_by_power = phase.get("orders", {})
        result_groups = phase.get("order_results", {})
        if not isinstance(orders_by_power, Mapping):
            continue
        for power in POWERS:
            orders = orders_by_power.get(power, [])
            if not isinstance(orders, list) or not orders:
                continue
            flattened_results: list[dict[str, str]] = []
            groups = result_groups.get(power, {}) if isinstance(result_groups, Mapping) else {}
            if isinstance(groups, Mapping):
                for order_type, rows in groups.items():
                    if not isinstance(rows, list):
                        continue
                    for result in rows:
                        if isinstance(result, Mapping):
                            flattened_results.append(
                                {
                                    "order": str(result.get("order", "")),
                                    "result": str(result.get("result", "")),
                                    "order_type": str(order_type),
                                }
                            )
            outcomes.append(
                {
                    "phase": phase_name,
                    "power": power,
                    "orders": [str(order) for order in orders],
                    "results": flattened_results,
                    "visibility": "public",
                }
            )
    return outcomes


def _load_diplomacy_validator(engine_root: Path) -> Any:
    if not (engine_root / "diplomacy" / "utils" / "export.py").is_file():
        raise ImportError("Diplomacy adjudicator root is incomplete")
    root_text = str(engine_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("diplomacy.utils.export")
    module_path = Path(str(module.__file__)).resolve()
    if not module_path.is_relative_to(engine_root):
        raise ImportError("Diplomacy adjudicator resolved outside the selected root")
    return module.is_valid_saved_game


def _native_dir(native_root: Path, source_path: Path, prefix: str) -> Path:
    stem = _gzip_jsonl_stem(source_path)
    if stem == "pilot-00000":
        return native_root / f"pilot-{prefix}"
    if stem.startswith("new-"):
        return native_root / f"{prefix}-{stem.removeprefix('new-')}"
    return native_root / f"{prefix}-{stem}"


def _find_centauri_dataset(native_root: Path, preferred: Path, episode_id: str) -> Path | None:
    candidates = [preferred]
    candidates.extend(
        path
        for path in sorted(native_root.glob("pilot-*/centauri/native-centauri.jsonl"))
        if path != preferred
    )
    matches = [path for path in candidates if path.is_file() and _jsonl_has_episode(path, episode_id)]
    return matches[0] if len(matches) == 1 else None


def _find_they_sing_log(native_root: Path, preferred_dir: Path, episode_id: str) -> Path | None:
    candidates = sorted(preferred_dir.glob("*.jsonl"))
    candidates.extend(
        path
        for path in sorted(native_root.glob(f"pilot-*/they-sing/{episode_id}.jsonl"))
        if path not in candidates
    )
    matches = [path for path in candidates if path.is_file() and _jsonl_has_session(path, episode_id)]
    return matches[0] if len(matches) == 1 else None


def _jsonl_has_episode(path: Path, episode_id: str) -> bool:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            return json.loads(line).get("episode_id") == episode_id
    return False


def _jsonl_has_session(path: Path, session_id: str) -> bool:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "session_created":
                return row.get("sessionId") == session_id
    return False


def _gzip_jsonl_stem(path: Path) -> str:
    name = path.name
    if not name.endswith(".jsonl.gz"):
        raise ValueError(f"expected .jsonl.gz source: {path.name}")
    return name[: -len(".jsonl.gz")]


def _replay_cache_key(game: str, source_path: Path, record: Mapping[str, Any]) -> tuple[str, str]:
    if game in {"centauri", "they_sing"}:
        return game, str(record.get("episode_id", ""))
    return game, str(source_path)


def _inconclusive(reason: str, artifacts: Sequence[str]) -> ReplayResult:
    return ReplayResult("INCONCLUSIVE", reason, {}, tuple(sorted(set(artifacts))), {})


def _existing_hashes(*paths: Path) -> tuple[str, ...]:
    return tuple(sorted({sha256_file(path) for path in paths if path.is_file()}))


def _run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(list(argv), 124, "", type(exc).__name__)


def _read_gzip_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _they_sing_turn_commitment(path: Path) -> str | None:
    if not path.is_file():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    completed = [
        {
            "turn": row.get("turn"),
            "post_state_hash": row.get("trace", {}).get("post_state_hash"),
        }
        for row in rows
        if row.get("type") == "turn_completed"
    ]
    return sha256_json(completed) if completed else None


def _unique_review_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    compared_fields = ("game", "item_id", "model_family", "reasoning_effort", "record_hash", "split")
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id:
            raise ValueError("review map item ID is missing")
        prior = unique.get(item_id)
        if prior is not None and any(prior.get(field) != row.get(field) for field in compared_fields):
            raise ValueError(f"reviewer maps disagree for item: {item_id}")
        unique[item_id] = row
    return unique


def _unique_by_item(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in unique:
            raise ValueError(f"{label} item IDs must be non-empty and unique")
        unique[item_id] = row
    return unique


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = "".join(
        json.dumps(to_jsonable(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )
    path.write_text(body, encoding="utf-8")


def _is_beneath(path: Path, roots: Sequence[Path]) -> bool:
    return any(path.is_relative_to(root) for root in roots)
