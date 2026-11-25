"""Collate pinned Model Atlas extracts into deduplicated discovery records, a coverage ledger and counts.

Offline and deterministic. Inputs are the repository queue (``repositories.json``) and the committed
extract files it pins by SHA-256. Optionally the curated LBF sources are collated alongside. Nothing
here reads weights, executes upstream code or promotes records to training.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datagoblin import model_atlas
from datagoblin.model_atlas import SCHEMA, canonical, sha256, write_jsonl
from datagoblin.model_atlas_extract import EXTRACT_SCHEMA

QUEUE_SCHEMA = "1.0"
QUEUE_STATUSES = ("extracted", "partial", "queued", "no_artifacts", "inaccessible", "excluded")
REFERENCE_PREFIXES = (
    "source_", "base_", "original_", "resume_", "initial_", "round0_", "parent_", "frozen_", "teacher_",
)
PARENT_PREFIXES = ("source_", "base_", "original_", "resume_", "parent_", "teacher_")
PARAM_KEYS = ("parameters", "params", "param_count", "n_params", "num_params", "parameter_count")
TRAINABLE_KEYS = ("trainable_params", "trainable_parameters", "match_params")
PASS_KEYS = ("passes", "loops", "n_loops", "loop_count", "recurrent_passes")
ARCH_KEYS = ("arch", "architecture", "arm", "model", "model_name", "name", "variant", "kind", "topology")
SEED_KEYS = ("seed", "train_seed", "training_seed")
RECIPE_KEYS = (
    "steps", "updates", "best_step", "epochs", "batch_size", "lr", "learning_rate", "final_learning_rate",
    "weight_decay", "optimizer", "lam", "lambda", "warmup", "dtype", "precision", "schedule", "lr_schedule",
    "device", "seconds", "hidden", "width", "depth", "d_model", "heads", "layers", "bits", "exposure", "rounds",
)
DENOMINATOR_KEYS = re.compile(r"(^n$|^n_|_n$|items|count|^num_|total|trajectories|instances|eval_)", re.I)
PARAM_BINS = (
    ("lt_1e4", 0, 10_000), ("1e4_1e5", 10_000, 100_000), ("1e5_1e6", 100_000, 1_000_000),
    ("1e6_1e7", 1_000_000, 10_000_000), ("1e7_1e8", 10_000_000, 100_000_000),
    ("1e8_1e9", 100_000_000, 1_000_000_000), ("ge_1e9", 1_000_000_000, 10**30),
)


SNAPSHOT_NAME = re.compile(r"(^|[_\-.])(step|epoch|iter|iteration|update)[_\-]?\d+", re.I)
SAE_HINT = re.compile(r"(^|[_\-/])(sae|dictionar)", re.I)


def artifact_kind(prim: list[dict[str, Any]]) -> str:
    """Classify what kind of artifact a digest names; used to avoid counting snapshots and SAEs as base models."""
    votes: Counter[str] = Counter()
    for p in prim:
        text = " ".join([p["receipt_path"], p["pointer"], *p.get("artifact_names", [])])
        keys = " ".join({**p["context"], **p["fields"]})
        if "transcoder" in text.lower():
            votes["transcoder"] += 1
        elif SAE_HINT.search(text) or "fvu" in keys or "dead_latent" in keys:
            votes["sae_dictionary"] += 1
        elif any(SNAPSHOT_NAME.search(n) for n in p.get("artifact_names", [])):
            votes["intermediate_snapshot"] += 1
        else:
            votes["final_or_primary"] += 1
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def is_reference_role(role: str) -> bool:
    return role.startswith(REFERENCE_PREFIXES)


def population_group(study_path: str) -> str:
    """Heuristic lineage group: drop version/recovery/attempt/ordinal suffixes from the study directory name."""
    name = study_path.rstrip("/").split("/")[-1] if "/results/" not in study_path else study_path.split("/")[-3]
    name = re.sub(r"(_recovery\d+|_attempt\d+|_invalid\w*)$", "", name)
    previous = None
    while previous != name:
        previous = name
        name = re.sub(r"_v\d+(?:_\d+)*$", "", name)
    name = re.sub(r"_?\d+[a-z]?$", "", name)
    match = re.match(r"^([a-z]+)\d+[a-z]?_", name)
    if match:
        name = match.group(1)
    return name or study_path.replace("/", "_")


def _blob_uri(repo: str, commit: str, path: str, tree: bool = False) -> str:
    return f"https://github.com/{repo}/{'tree' if tree else 'blob'}/{commit}/{path}"


def _first(fields: dict[str, Any], keys: tuple[str, ...], kind: type | tuple[type, ...]) -> Any:
    for key in keys:
        value = fields.get(key)
        if isinstance(value, kind) and type(value) is not bool:
            return value
    return None


def _mode(values: list[Any]) -> tuple[Any, list[Any]]:
    if not values:
        return None, []
    counts = Counter(json.dumps(v, sort_keys=True) for v in values)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    distinct = [json.loads(k) for k, _ in best]
    return distinct[0], distinct


def load_queue(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_bytes())
    if cfg.get("schema_version") != QUEUE_SCHEMA or not isinstance(cfg.get("repositories"), list):
        raise ValueError("unknown repository queue schema")
    seen = set()
    for entry in cfg["repositories"]:
        repo = entry["repository"]
        if repo in seen:
            raise ValueError(f"duplicate queue entry {repo}")
        seen.add(repo)
        if entry["status"] not in QUEUE_STATUSES:
            raise ValueError(f"unknown queue status {entry['status']}")
        for ex in entry.get("extracts", []):
            if not model_atlas.HEX40.fullmatch(ex["commit"]) or not model_atlas.HEX64.fullmatch(ex["sha256"]):
                raise ValueError("extract pins must be a commit SHA-1 and file SHA-256")
    return cfg


def load_extracts(queue: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    extracts = []
    for entry in queue["repositories"]:
        for pin in entry.get("extracts", []):
            data = (base / pin["file"]).read_bytes()
            if sha256(data) != pin["sha256"]:
                raise ValueError(f"extract digest mismatch: {pin['file']}")
            ex = json.loads(data)
            agrees = ex["commit"] == pin["commit"] and ex["repository"] == entry["repository"]
            if ex["schema"] != EXTRACT_SCHEMA or not agrees:
                raise ValueError(f"extract pin disagrees with content: {pin['file']}")
            if ex.get("weights_loaded") != 0 or ex.get("weight_bytes_read") != 0 or ex.get("upstream_code_executed"):
                raise ValueError("extract violates the no-weights, no-execution contract")
            extracts.append(ex)
    return extracts


def _metrics(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    skip = {*PARAM_KEYS, *TRAINABLE_KEYS, *PASS_KEYS, *ARCH_KEYS, *SEED_KEYS, *RECIPE_KEYS, "bytes", "size"}
    metrics: dict[str, Any] = {}
    denominators: dict[str, Any] = {}
    for key, value in fields.items():
        if key in skip or isinstance(value, str):
            continue
        (denominators if DENOMINATOR_KEYS.search(key) and type(value) is int else metrics)[key] = value
    return metrics, denominators


GENERIC_SEGMENTS = frozenset({
    "models", "rows", "fits", "checkpoints", "training", "selected", "results", "evidence", "arms", "snapshot", "key",
    "terminal_checkpoint", "eval_table", "training_cells", "summary", "runs", "cells", "items", "entries",
})


def _label_from_pointer(pointer: str) -> str | None:
    parts = [re.sub(r"\[\d+\]$", "", p) for p in pointer.strip("/").split("/")]
    for candidate in reversed([p for p in parts if p][:-1]):
        plural = candidate.endswith(("s", "table", "cells")) and not re.search(r"\d", candidate)
        if candidate in GENERIC_SEGMENTS or plural:
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]{0,47}", candidate):
            return candidate
    return None


class Corpus:
    def __init__(self) -> None:
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.studies: dict[str, dict[str, Any]] = {}
        self.weights: dict[str, dict[str, Any]] = {}
        self.siblings: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)


def collate_extracts(extracts: list[dict[str, Any]]) -> Corpus:
    corpus = Corpus()
    repo_commit: dict[str, str] = {}
    for ex in extracts:
        repo = ex["repository"]
        repo_commit[f"{repo}@{ex['commit']}"] = repo
        short = repo.split("/")[-1]
        for study_path, st in ex["studies"].items():
            sid = f"{short}:{study_path}"
            if sid in corpus.studies:
                continue
            corpus.studies[sid] = {"study_path": study_path, "repository": repo, "commit": ex["commit"],
                                   "short": short, "info": st, "checkpoint_ids": set(), "blob_ids": set()}
            for w in st["weights_like"]:
                rec = corpus.weights.setdefault(w["git_blob_sha1"], {"size": w["size"], "locations": []})
                rec["locations"].append({"repository": repo, "commit": ex["commit"], "path": w["path"]})
        for obs in ex["observations"]:
            entry = corpus.checkpoints.setdefault(obs["sha256"], {"primary": [], "reference": [], "repos": set()})
            item = {**obs, "repository": repo, "commit": ex["commit"], "short": short}
            entry["reference" if is_reference_role(obs["role"]) else "primary"].append(item)
            entry["repos"].add(repo)
            corpus.siblings[(repo, ex["commit"], obs["receipt_path"], obs["pointer"].rsplit("/", 1)[0])].append(item)
            for ref in obs.get("ancestor_refs", []):
                held = corpus.checkpoints.setdefault(ref["sha256"], {"primary": [], "reference": [], "repos": set()})
                held["reference"].append({**item, "sha256": ref["sha256"], "role": ref["role"], "fields": {},
                                          "context": {}, "ancestor_refs": [], "artifact_names": []})
                held["repos"].add(repo)
    for sha, entry in corpus.checkpoints.items():
        for item in [*entry["primary"], *entry["reference"]]:
            sid = f"{item['short']}:{item['study']}"
            if sid in corpus.studies:
                corpus.studies[sid]["checkpoint_ids"].add(sha)
    return corpus


def _receipt_tags(ex_studies: dict[str, Any], study_path: str, receipt_path: str) -> list[str]:
    for r in ex_studies.get(study_path, {}).get("receipts", []):
        if r["path"] == receipt_path:
            return list(r["evidence_tags"])
    return []


def build_models(corpus: Corpus, extracts: list[dict[str, Any]], curated_ids: frozenset[str]) -> tuple[
        list[dict[str, Any]], dict[str, Any]]:
    by_repo_commit = {(ex["repository"], ex["commit"]): ex for ex in extracts}
    blob_for_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for blob, w in corpus.weights.items():
        for loc in w["locations"]:
            blob_for_name[(loc["repository"], loc["path"].rsplit("/", 1)[-1])].append(
                {"blob": blob, **loc, "size": w["size"]})
    models: list[dict[str, Any]] = []
    claimed_blobs: set[str] = set()
    referenced_only: list[dict[str, Any]] = []
    already_curated = 0
    for sha in sorted(corpus.checkpoints):
        entry = corpus.checkpoints[sha]
        if not entry["primary"]:
            ref = entry["reference"][0]
            referenced_only.append({"checkpoint_sha256": sha, "first_role": ref["role"],
                                    "receipt_uri": _blob_uri(ref["repository"], ref["commit"], ref["receipt_path"]),
                                    "study": f"{ref['short']}:{ref['study']}"})
            continue
        if f"sha256:{sha}" in curated_ids:
            already_curated += 1
            continue
        prim = sorted(entry["primary"], key=lambda o: (o["receipt_path"], o["pointer"], o["role"]))
        anchor = prim[0]
        repo, commit, short = anchor["repository"], anchor["commit"], anchor["short"]
        fields_all = [{**p["context"], **p["fields"]} for p in prim]
        trainable, _ = _mode([v for f in fields_all if (v := _first(f, TRAINABLE_KEYS[:2], int)) is not None])
        params, params_all = _mode([v for f in fields_all if (v := _first(f, PARAM_KEYS, int)) is not None])
        passes, passes_all = _mode([v for f in fields_all if (v := _first(f, PASS_KEYS, int)) is not None])
        seeds = sorted({v for f in fields_all if (v := _first(f, SEED_KEYS, int)) is not None})
        arch_vals = [str(v) for f in fields_all if (v := _first(f, ARCH_KEYS, str)) is not None]
        label = _mode(arch_vals)[0] or next(
            (lab for p in prim if (lab := _label_from_pointer(p["pointer"]))), None)
        recipe = {k: v for f in fields_all for k, v in f.items() if k in RECIPE_KEYS}
        if params is not None and params < 1:
            params = None
        study_ids = sorted({f"{p['short']}:{p['study']}" for p in [*prim, *entry["reference"]]})
        anchor_study = f"{anchor['short']}:{anchor['study']}"
        group = f"{short}:{population_group(anchor['study'])}"
        # archive reference: a tracked weight file of the same basename, never read
        names = sorted({n for p in prim for n in p.get("artifact_names", [])})
        archive = None
        for name in names:
            options = sorted(blob_for_name.get((repo, name), []), key=lambda o: o["path"])
            if options:
                best = options[0]
                archive = {"git_blob_sha1": best["blob"], "path": best["path"], "size": best["size"],
                           "uri": _blob_uri(best["repository"], best["commit"], best["path"])}
                claimed_blobs.add(best["blob"])
                break
        quant = _mode([str(f["quant"]) for f in fields_all if "quant" in f])[0]
        base_ids = sorted({v for p in prim for v in p.get("model_ids", {}).values()})
        phenotype = []
        mech: dict[str, set[str]] = {"mechanistic": set(), "causal": set(), "formal": set()}
        receipts_seen: set[str] = set()
        for p in prim:
            ex = by_repo_commit[(p["repository"], p["commit"])]
            tags = _receipt_tags(ex["studies"], p["study"], p["receipt_path"])
            uri = _blob_uri(p["repository"], p["commit"], p["receipt_path"])
            for tag in tags:
                if tag in mech:
                    mech[tag].add(uri)
            metrics, denominators = _metrics(p["fields"])
            receipts_seen.add(uri)
            if metrics or denominators:
                phenotype.append({
                    "study_id": f"{p['short']}:{p['study']}", "scope": "receipt_observation", "split": "unspecified",
                    "evidence": "source_reported", "independent_verification": False,
                    "source_uri": uri, "pointer": p["pointer"], "metrics": metrics, "denominators": denominators,
                    "metric_definitions": None, "verdicts": p.get("verdicts", {}),
                })
            sib = p.get("sibling_results")
            if sib:
                phenotype.append({
                    "study_id": f"{p['short']}:{p['study']}", "scope": "receipt_observation", "split": "unspecified",
                    "evidence": "source_reported", "independent_verification": False,
                    "source_uri": _blob_uri(p["repository"], p["commit"], sib["source_path"]),
                    "pointer": "/per_loop", "metrics": sib["metrics"], "denominators": sib["denominators"],
                    "metric_definitions": None, "verdicts": {},
                })
        phenotype.sort(key=lambda r: (r["source_uri"], r["pointer"]))
        near = [r for p in prim for r in corpus.siblings[(p["repository"], p["commit"], p["receipt_path"],
                                                           p["pointer"].rsplit("/", 1)[0])]]
        ancestors = [a for p in prim for a in p.get("ancestor_refs", [])]
        parents = sorted({(r["sha256"], r["role"]) for r in [*near, *ancestors]
                          if r["sha256"] != sha and r["role"].startswith(PARENT_PREFIXES)})
        # a parent edge is asserted only for the checkpoint's own receipt dict, not for every co-located reference
        notation_arch = label or "unknown"
        arch_source = "receipt_field" if arch_vals else ("receipt_pointer" if label else "unknown")
        notation = (f"MA1[{short}/{anchor['study'].replace('/', '.')}/{notation_arch}@s"
                    f"{seeds[0] if seeds else 'null'};P={params or 'null'};K={passes or 'null'}]")
        models.append({
            "schema_version": SCHEMA, "record_type": "model", "model_id": f"sha256:{sha}", "notation": notation,
            "family": group, "population_group": group, "study_id": anchor_study,
            "genotype": {"architecture": notation_arch, "parameter_count": params, "passes": passes,
                         "implementation_uri": None,
                         "architecture_source": arch_source,
                         "conflicting_parameter_counts": params_all[1:], "conflicting_pass_counts": passes_all[1:],
                         "quantization": quant, "recurrence_topology": None,
                         "trainable_parameter_count": trainable, "base_model_ids": base_ids},
            "training_recipe": {"declared": recipe or None, "training_seed": seeds[0] if seeds else None,
                                "training_seed_candidates": seeds[1:], "reported_fit_receipt": None,
                                "extraction": "generic_receipt_scan"},
            "phenotype": phenotype,
            "mechanistic_profile": {"study_ref": anchor_study, "scope": "receipt_tags_not_verified",
                                    "receipts_with_tag": sorted(mech["mechanistic"])[:8],
                                    "receipt_count_with_tag": len(mech["mechanistic"])},
            "causal_evidence": {"study_ref": anchor_study, "scope": "receipt_tags_not_verified",
                                "receipts_with_tag": sorted(mech["causal"])[:8],
                                "receipt_count_with_tag": len(mech["causal"])},
            "formal_properties": {"study_ref": anchor_study, "scope": "receipt_tags_not_verified",
                                  "receipts_with_tag": sorted(mech["formal"])[:8],
                                  "receipt_count_with_tag": len(mech["formal"])},
            "lineage": {"architecture_predecessor_study": None,
                        "weight_parent": parents[0][0] if len(parents) == 1 else None,
                        "weight_parent_candidates": [{"sha256": s, "role": r} for s, r in parents],
                        "group_id": group, "observed_in_studies": study_ids},
            "checkpoint": {"reported_sha256": sha, "verified_bytes": False,
                           "archive_directory_uri": archive["uri"].rsplit("/", 1)[0] if archive else None,
                           "tracked_weight_file": archive, "artifact_kind": artifact_kind(prim),
                           "receipt_uris": sorted(receipts_seen)[:12], "receipt_count": len(receipts_seen)},
            "provenance": {"repository": repo, "commit": commit, "evidence": "source_reported",
                           "extract_schema": EXTRACT_SCHEMA, "roles_observed": sorted({p["role"] for p in prim}),
                           "repositories_observed": sorted(entry["repos"])},
            "training_ready": False, "data_rights_review": "pending", "data_split": "unassigned",
            "custody_lane": "discovery",
        })
    # tracked weight files whose digests no receipt names: identity is the Git blob SHA-1 only
    for blob in sorted(set(corpus.weights) - claimed_blobs):
        w = corpus.weights[blob]
        loc = sorted(w["locations"], key=lambda x: (x["repository"], x["path"]))
        first = loc[0]
        short = first["repository"].split("/")[-1]
        study = next((sid for sid, s in corpus.studies.items()
                      if s["repository"] == first["repository"] and first["path"].startswith(s["study_path"] + "/")),
                     f"{short}:.")
        group = f"{short}:{population_group(study.split(':', 1)[1])}"
        models.append({
            "schema_version": SCHEMA, "record_type": "model", "model_id": f"gitblob:{blob}",
            "notation": f"MA1[{short}/{study.split(':', 1)[1].replace('/', '.')}/unknown@snull;P=null;K=null]",
            "family": group, "population_group": group, "study_id": study,
            "genotype": {"architecture": "unknown", "parameter_count": None, "passes": None,
                         "implementation_uri": None, "architecture_source": "unknown",
                         "conflicting_parameter_counts": [], "conflicting_pass_counts": [],
                         "quantization": None, "recurrence_topology": None},
            "training_recipe": {"declared": None, "training_seed": None, "training_seed_candidates": [],
                                "reported_fit_receipt": None, "extraction": "tracked_weight_file_only"},
            "phenotype": [],
            "mechanistic_profile": {"study_ref": study, "scope": "none_collected"},
            "causal_evidence": {"study_ref": study, "scope": "none_collected"},
            "formal_properties": {"study_ref": study, "scope": "none_collected"},
            "lineage": {"architecture_predecessor_study": None, "weight_parent": None,
                        "weight_parent_candidates": [], "group_id": group, "observed_in_studies": [study]},
            "checkpoint": {"reported_sha256": None, "verified_bytes": False, "archive_directory_uri": None,
                           "git_blob_sha1": blob, "size": w["size"],
                           "locations": [{**x, "uri": _blob_uri(x["repository"], x["commit"], x["path"])}
                                         for x in loc[:6]], "location_count": len(loc),
                           "tracked_weight_file": None, "artifact_kind": "unclassified_tracked_file",
                           "receipt_uris": [], "receipt_count": 0},
            "provenance": {"repository": first["repository"], "commit": first["commit"],
                           "evidence": "tree_entry_only", "extract_schema": EXTRACT_SCHEMA,
                           "roles_observed": [], "repositories_observed": sorted({x["repository"] for x in loc})},
            "training_ready": False, "data_rights_review": "pending", "data_split": "unassigned",
            "custody_lane": "discovery",
        })
    stats = {"referenced_only": referenced_only, "already_curated": already_curated}
    return sorted(models, key=lambda m: (m["notation"], m["model_id"])), stats


def _curated_dir(study: dict[str, Any]) -> str:
    uri = study["provenance"]["manifest_uri"]
    return uri.split(f"/blob/{study['provenance']['commit']}/", 1)[1].rsplit("/", 1)[0]


def build_studies(corpus: Corpus, models: list[dict[str, Any]], curated_studies: list[dict[str, Any]],
                  curated_dirs: set[str]) -> list[dict[str, Any]]:
    per_study_models: dict[str, list[str]] = defaultdict(list)
    for m in models:
        for sid in m["lineage"]["observed_in_studies"]:
            per_study_models[sid].append(m["model_id"])
    rows = []
    for sid, s in sorted(corpus.studies.items()):
        info = s["info"]
        repo, commit = s["repository"], s["commit"]
        if not (info["receipts"] or info["weights_like"] or info["docs"]) or s["study_path"] in curated_dirs:
            continue
        group = f"{s['short']}:{population_group(s['study_path'])}"
        rows.append({
            "schema_version": SCHEMA, "record_type": "study", "study_id": sid, "family": group,
            "population_group": group, "split": "unspecified", "independent_roots": None, "reference_budget": None,
            "metrics_scope": "receipt_level_not_aggregated", "stats": {},
            "interpretation_notes": [],
            "evidence_tags": info["evidence_tags"],
            "negative_or_correction_docs": [_blob_uri(repo, commit, p) for p in info["negative_or_correction_docs"]],
            "report_uris": [_blob_uri(repo, commit, d["path"]) for d in info["docs"][:20]],
            "report_count": len(info["docs"]),
            "receipts": [{"uri": _blob_uri(repo, commit, r["path"]), "git_blob_sha1": r["git_blob_sha1"],
                          "checkpoint_observations": r["observations"], "evidence_tags": r["evidence_tags"],
                          **({"schema": r["schema"]} if "schema" in r else {}),
                          **({"top_level_verdicts": r["top_level_verdicts"]} if "top_level_verdicts" in r else {})}
                         for r in info["receipts"][:200]],
            "receipt_count": len(info["receipts"]),
            "checkpoint_identity_count": len(per_study_models.get(sid, [])),
            "tracked_weight_files": len(info["weights_like"]),
            "jsonl_not_collected": info["jsonl_not_collected"],
            "mechanistic_artifact_uris": [_blob_uri(repo, commit, w["path"]) for w in info["weights_like"][:6]],
            "provenance": {"repository": repo, "commit": commit, "evidence": "source_reported"},
            "training_ready": False, "data_rights_review": "pending", "data_split": "unassigned",
            "custody_lane": "discovery",
        })
    for st in curated_studies:
        st["provenance"].setdefault("repository", "MoralityLabAI/LoopedTransformers")
        st.setdefault("evidence_tags", [])
        st.setdefault("receipts", [])
        st.setdefault("receipt_count", 1)
        st.setdefault("report_count", 1)
        st.setdefault("checkpoint_identity_count", len(per_study_models.get(st["study_id"], [])))
        st.setdefault("tracked_weight_files", 0)
        st.setdefault("jsonl_not_collected", 0)
    return sorted([*curated_studies, *rows], key=lambda r: r["study_id"])


def bin_params(value: int | None) -> str:
    if value is None:
        return "unknown"
    for name, lo, hi in PARAM_BINS:
        if lo <= value < hi:
            return name
    return "unknown"


def has_evidence(block: dict[str, Any]) -> bool:
    if block.get("receipt_count_with_tag", 0) > 0:
        return True
    return block.get("scope") in ("study_aggregate_not_seed_specific", "reported_theorems_and_certificate_receipts")


def inventory(models: list[dict[str, Any]], studies: list[dict[str, Any]], stats: dict[str, Any],
              extracts: list[dict[str, Any]]) -> dict[str, Any]:
    def repo_of(m: dict[str, Any]) -> str:
        return str(m["provenance"].get("repository", "unknown"))

    by_repo = Counter(repo_of(m) for m in models)
    by_arch = Counter(m["genotype"]["architecture"] for m in models)
    by_group = Counter(m["population_group"] for m in models)
    def kind(m: dict[str, Any]) -> str:
        return str(m["checkpoint"].get("artifact_kind", "final_or_primary"))

    primary = [m for m in models if kind(m) == "final_or_primary"]
    cells = {(m["population_group"], m["genotype"]["architecture"], m["genotype"]["parameter_count"]) for m in primary}
    seed_collapsed = {(m["population_group"], m["genotype"]["architecture"]) for m in primary}
    receipts = {r["uri"] for s in studies for r in s.get("receipts", [])}
    receipts |= {u for m in models for u in m["checkpoint"].get("receipt_uris", [])}
    multi_study = sum(1 for m in models if len(m["lineage"]["observed_in_studies"]) > 1)
    mech = Counter()
    for key, field in (("mechanistic", "mechanistic_profile"), ("causal", "causal_evidence"),
                       ("formal", "formal_properties")):
        mech[key] = sum(has_evidence(m[field]) for m in models)
    evidence = {
        "with_parameter_count": sum(m["genotype"]["parameter_count"] is not None for m in models),
        "with_training_seed": sum(m["training_recipe"]["training_seed"] is not None for m in models),
        "with_declared_recipe": sum(bool(m["training_recipe"]["declared"]) for m in models),
        "with_phenotype_rows": sum(bool(m["phenotype"]) for m in models),
        "with_known_architecture": sum(m["genotype"]["architecture"] != "unknown" for m in models),
        "with_tracked_weight_file_reference": sum(
            bool(m["checkpoint"].get("tracked_weight_file") or m["checkpoint"].get("git_blob_sha1")) for m in models),
        "with_weight_parent": sum(m["lineage"]["weight_parent"] is not None for m in models),
        "tree_entry_only": sum(m["provenance"]["evidence"] == "tree_entry_only" for m in models),
    }
    docs_bytes = sum(len(canonical(r)) for r in [*models, *studies])
    return {
        "schema_version": SCHEMA, "model_count": len(models), "study_count": len(studies),
        "candidate_document_count": len(models) + len(studies), "candidate_document_json_bytes": docs_bytes,
        "training_ready": False, "training_examples_released": 0, "weights_loaded": 0, "weight_bytes_read": 0,
        "upstream_code_executed": False,
        "unique_corpus": {
            "distinct_checkpoint_identities": len(models),
            "identities_by_artifact_kind": dict(sorted(Counter(kind(m) for m in models).items())),
            "primary_model_identities": len(primary),
            "distinct_design_cells_ignoring_seed_primary_only": len(cells),
            "distinct_group_architecture_pairs_primary_only": len(seed_collapsed),
            "distinct_population_groups": len(by_group),
            "distinct_receipt_blobs_referenced": len(receipts),
            "models_observed_in_multiple_studies": multi_study,
            "already_curated_and_skipped": stats["already_curated"],
            "referenced_only_identities": len(stats["referenced_only"]),
            "note": "Seed replicates and re-reports of one checkpoint are counted once; "
                    "shared reports live at study level.",
        },
        "model_count_by_repository": dict(sorted(by_repo.items())),
        "model_count_by_architecture_top40": dict(by_arch.most_common(40)),
        "model_count_by_population_group": dict(sorted(by_group.items(), key=lambda kv: (-kv[1], kv[0]))),
        "model_count_by_parameter_range": dict(Counter(bin_params(m["genotype"]["parameter_count"]) for m in models)),
        "evidence_availability": evidence,
        "mechanistic_coverage_receipt_tags": {k: mech[k] for k in ("mechanistic", "causal", "formal")},
        "extract_pins": sorted(f"{ex['repository']}@{ex['commit']}" for ex in extracts),
    }


def ledger(queue: dict[str, Any], extracts: list[dict[str, Any]], models: list[dict[str, Any]],
           studies: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    ex_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in extracts:
        ex_by_repo[ex["repository"]].append(ex)
    model_by_repo = Counter(str(m["provenance"].get("repository")) for m in models)
    ref_by_repo = Counter(r["study"].split(":", 1)[0] for r in stats["referenced_only"])
    rows = []
    for entry in queue["repositories"]:
        repo = entry["repository"]
        exs = ex_by_repo.get(repo, [])
        study_rows = [s for s in studies if s["provenance"].get("repository") == repo]
        no_receipt = sorted(s["study_id"] for s in study_rows
                            if s["checkpoint_identity_count"] == 0 and s["report_count"] > 0)
        rows.append({
            "repository": repo, "priority": entry["priority"], "status": entry["status"],
            "pinned_commits": sorted({e["commit"] for e in exs}),
            "tree_files_scanned": sum(e["tree_files"] for e in exs),
            "studies_with_records": len(study_rows),
            "receipt_files": sum(s["receipt_count"] for s in study_rows),
            "checkpoint_observations": sum(len(e["observations"]) for e in exs),
            "model_records": model_by_repo.get(repo, 0),
            "tracked_weight_files": sum(s["tracked_weight_files"] for s in study_rows),
            "referenced_only_identities": ref_by_repo.get(repo.split("/")[-1], 0),
            "jsonl_not_collected": sum(s["jsonl_not_collected"] for s in study_rows),
            "excluded_by_custody": dict(sorted(Counter(
                {k: v for e in exs for k, v in e["excluded_by_custody"].items()}).items())),
            "studies_with_reports_but_no_checkpoint_receipts": no_receipt[:40],
            "studies_with_reports_but_no_checkpoint_receipts_count": len(no_receipt),
            "notes": entry.get("notes", []),
            "unresolved_provenance": entry.get("unresolved_provenance", []),
            "missing_artifacts": entry.get("missing_artifacts", []),
        })
    return {"schema_version": SCHEMA, "repositories": rows,
            "totals": {"repositories": len(rows), "by_status": dict(Counter(r["status"] for r in rows))}}


def render_ledger(led: dict[str, Any], inv: dict[str, Any]) -> str:
    lines = ["# Model Atlas repository coverage ledger", "",
             "Generated by `python -m datagoblin.model_atlas_corpus`; do not edit by hand.",
             "All records are discovery-only.",
             "", f"Unique checkpoint identities: **{inv['unique_corpus']['distinct_checkpoint_identities']}**; "
             f"primary models: **{inv['unique_corpus']['primary_model_identities']}**; design cells ignoring seed: "
             f"**{inv['unique_corpus']['distinct_design_cells_ignoring_seed_primary_only']}**; "
             f"study records: **{inv['study_count']}**.", "",
             "| Priority | Repository | Status | Commit | Studies | Receipts | Models | Tracked weights | Ref-only |",
             "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(led["repositories"], key=lambda x: (x["priority"], x["repository"])):
        commit = ", ".join(c[:10] for c in r["pinned_commits"]) or "n/a"
        lines.append(f"| {r['priority']} | {r['repository']} | {r['status']} | {commit} | "
                     f"{r['studies_with_records']} | "
                     f"{r['receipt_files']} | {r['model_records']} | {r['tracked_weight_files']} | "
                     f"{r['referenced_only_identities']} |")
    lines += ["", "## Open items", ""]
    for r in sorted(led["repositories"], key=lambda x: (x["priority"], x["repository"])):
        items = [*r["unresolved_provenance"], *r["missing_artifacts"], *r["notes"]]
        if r["jsonl_not_collected"]:
            items.append(f"{r['jsonl_not_collected']} JSONL files (episode rows and traces) are not collected.")
        if r["studies_with_reports_but_no_checkpoint_receipts_count"]:
            items.append(f"{r['studies_with_reports_but_no_checkpoint_receipts_count']} study directories have reports "
                         "but no machine-readable checkpoint digests.")
        if items:
            lines += [f"### {r['repository']}", "", *[f"- {i}" for i in items], ""]
    return "\n".join(lines) + "\n"


def emit_corpus(out: Path, models: list[dict[str, Any]], studies: list[dict[str, Any]], inv: dict[str, Any],
                led: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    documents = [{"schema_version": SCHEMA, "document_id": r.get("model_id", r["study_id"]),
                  "population_group": r["population_group"], "custody_lane": "discovery", "training_ready": False,
                  "data_rights_review": "pending", "data_split": "unassigned", "text": canonical(r).decode("utf-8")}
                 for r in [*models, *studies]]
    hashes = {"models.jsonl": write_jsonl(out / "models.jsonl", models),
              "studies.jsonl": write_jsonl(out / "studies.jsonl", studies),
              "candidate_documents.jsonl": write_jsonl(out / "candidate_documents.jsonl", documents)}
    inv = {**inv, "output_sha256": hashes}
    (out / "inventory.json").write_bytes(canonical(inv) + b"\n")
    (out / "coverage_ledger.json").write_bytes(canonical(led) + b"\n")
    (out / "COVERAGE_LEDGER.md").write_text(render_ledger(led, inv), encoding="utf-8", newline="\n")
    return inv


def run(queue_path: Path, out: Path, sources: Path | None = None, cache: Path | None = None) -> dict[str, Any]:
    queue = load_queue(queue_path)
    extracts = load_extracts(queue, queue_path.parent)
    curated_models: list[dict[str, Any]] = []
    curated_studies: list[dict[str, Any]] = []
    if sources is not None and cache is not None:
        curated_models, curated_studies = model_atlas.collate(model_atlas.load_sources(sources), cache)
    curated_ids = frozenset(m["model_id"] for m in curated_models)
    corpus = collate_extracts(extracts)
    models, stats = build_models(corpus, extracts, curated_ids)
    for cm in curated_models:
        cm["lineage"].setdefault("observed_in_studies", [cm["study_id"]])
    all_models = sorted([*curated_models, *models], key=lambda m: (m["notation"], m["model_id"]))
    curated_dirs = {_curated_dir(st) for st in curated_studies}
    studies = build_studies(corpus, all_models, curated_studies, curated_dirs)
    inv = inventory(all_models, studies, stats, extracts)
    led = ledger(queue, extracts, all_models, studies, stats)
    return emit_corpus(out, all_models, studies, inv, led)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path, help="training/model_atlas/repositories.json")
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--sources", type=Path, help="curated sources.json (LBF fit manifests)")
    parser.add_argument("--cache", type=Path, help="cache of pinned curated manifests")
    args = parser.parse_args()
    print(json.dumps(run(args.queue, args.out, args.sources, args.cache), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
