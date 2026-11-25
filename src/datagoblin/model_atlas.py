"""Offline-first, metadata-only model atlas for Aleph research data.

Run python -m datagoblin.model_atlas --help. Does not deserialize checkpoints,
execute upstream code, read raw trajectories, or promote data into training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "1.0"
MAX_BYTES = 8 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
BUDGETS = {"16", "32", "64", "96", "128"}
FIT_FIELDS = {
    "steps", "batch_size", "fit_wall_clock_sec", "initial", "final",
    "shared_initial", "draws", "draws_sha256", "trace", "trace_sha256",
}
STUDY_FIELDS = (
    "rates", "costs", "budget_curves", "budget_curve_mean_closure_auc",
    "paired", "screens", "primary_screen", "memory_screen",
    "training", "probe_summary", "fresco_probe_summary",
    "abstraction_summary_calibration", "policy_certificates", "reference_certificates",
    "lean_theory_formalization", "native_online_tactic_calls",
)
PHASE_FIELDS = (
    "calibration_root_contrasts", "factorial_2x2", "equivalence_verification",
    "lean18_formalization", "calibration_new_lean_certificates",
    "isolated_clean_process_rss_mb", "synthetic_successor_cost_sweep",
    "adaptive_pass_calibration",
)
FIT_HASH_FIELDS = {
    "checkpoint_sha256", "initial", "final", "shared_initial", "draws",
    "draws_sha256", "trace", "trace_sha256",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    # Git object identity, not a substitute for cryptographic source review.
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _name(value: Any) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("invalid atlas identifier")
    return value


def _path(value: Any) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("invalid repository-relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or str(p) != value or ".." in p.parts:
        raise ValueError("unsafe repository-relative path")
    if not all(NAME.fullmatch(part) for part in p.parts):
        raise ValueError("unsafe repository path component")
    return value


def _metadata_path(value: Any) -> str:
    path = _path(value)
    # Only named metadata manifests; no weights, episodes, suites or raw traces.
    allowed = {"manifest.json", "calibration_phase_summary.json", "model_fit_manifest.json"}
    if PurePosixPath(path).name not in allowed:
        raise ValueError("manifest path is not on the metadata allowlist")
    for part in PurePosixPath(path).parts:
        if any(x in part.lower() for x in ("sealed", "holdout", "private", "ordinary", "ood", "test_split")):
            raise ValueError("evaluation/private path prohibited")
    return path


def validate_source(source: dict[str, Any]) -> None:
    for key in ("source_id", "study_id", "family"):
        _name(source[key])
    if not REPO.fullmatch(source["repository"]):
        raise ValueError("invalid repository")
    if not HEX40.fullmatch(source["commit"]) or not HEX40.fullmatch(source["manifest_git_blob_sha1"]):
        raise ValueError("source must be pinned to commit and exact Git blob")
    _metadata_path(source["manifest_path"])
    if source["scope"] != "open_development_summary":
        raise ValueError("only open development summaries are accepted")
    if source["adapter"] not in ("fit_manifest", "lbf18_calibration"):
        raise ValueError("unknown adapter")
    if type(source["independent_roots"]) is not int or source["independent_roots"] < 1:
        raise ValueError("independent root count required")
    if type(source["reference_budget"]) is not int or source["reference_budget"] < 1:
        raise ValueError("reference budget required")
    for key in ("registration_path", "report_path", "implementation_path"):
        _path(source[key])
    for path in source.get("mechanistic_artifact_paths", []):
        _path(path)


def load_sources(path: Path) -> list[dict[str, Any]]:
    cfg = json.loads(path.read_bytes())
    if cfg["schema_version"] != SCHEMA or not isinstance(cfg["sources"], list):
        raise ValueError("unknown source catalog schema")
    sources = cfg["sources"]
    for src in sources:
        validate_source(src)
    for key in ("source_id", "study_id"):
        if len({s[key] for s in sources}) != len(sources):
            raise ValueError(f"duplicate {key}")
    return sources


def _read_bounded(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("symlink sources are prohibited")
    with path.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("metadata exceeds byte limit")
    return data


def verify_manifest(src: dict[str, Any], data: bytes) -> dict[str, Any]:
    if len(data) > MAX_BYTES or git_blob_sha1(data) != src["manifest_git_blob_sha1"]:
        raise ValueError(f"{src['source_id']}: metadata blob mismatch")
    obj = json.loads(data)
    if obj["schema"] != src["manifest_schema"]:
        raise ValueError("unexpected upstream manifest schema")
    # A manifest may advertise a procedural seal, but this is not proof of custody.
    if obj.get("sealed_outcomes_opened") is not False:
        raise ValueError("manifest does not declare open-development-only scope")
    return obj


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Fetch only the immutable, explicitly configured raw GitHub metadata URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirects prohibited for metadata intake")


def fetch_sources(sources: list[dict[str, Any]], cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(NoRedirect())
    for src in sources:
        target = cache / f"{src['source_id']}.json"
        if target.exists():
            verify_manifest(src, _read_bounded(target))
            continue
        url = f"https://raw.githubusercontent.com/{src['repository']}/{src['commit']}/{src['manifest_path']}"
        request = urllib.request.Request(url, headers={"User-Agent": "DataGoblin-model-atlas/1"})
        with opener.open(request, timeout=30) as handle:
            data = handle.read(MAX_BYTES + 1)
        verify_manifest(src, data)
        with target.open("xb") as handle:
            handle.write(data)


def uri(src: dict[str, Any], path: str) -> str:
    return f"https://github.com/{src['repository']}/blob/{src['commit']}/{_path(path)}"


def _summary(value: Any, depth: int = 0) -> Any:
    """Only numeric/Boolean aggregate values, with bounded named keys.

    Never copy text, per-root arrays, labels, witness paths, seeds, or raw content.
    Numeric lists of length <= 2 retain intervals; longer arrays are omitted.
    """
    if depth > 6:
        return None
    if type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not NAME.fullmatch(key):
                continue
            low = key.lower()
            if low.startswith("_") or (
                low.startswith(("train_", "calibration_", "ordinary_", "ood_"))
                and re.search(r"_[0-9]{3}_", low)
            ):
                continue
            if any(x in low for x in ("witness", "secret", "token", "label", "generator_seed", "sealed",
                                      "root_ids", "lineage_ids", "checkpoint", "sha256", "path")):
                continue
            v = _summary(item, depth + 1)
            if v is not None:
                out[key] = v
        return out
    if (
        isinstance(value, list)
        and 1 <= len(value) <= 2
        and all(type(x) in (int, float) and math.isfinite(x) for x in value)
    ):
        return value
    return None


def _fit_receipt(fit: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for k in FIT_FIELDS:
        if k not in fit:
            continue
        v = fit[k]
        if k in FIT_HASH_FIELDS:
            if isinstance(v, str) and HEX64.fullmatch(v):
                result[k] = v
            else:
                raise ValueError("malformed fit hash")
        elif type(v) in (int, float) and math.isfinite(v) and v >= 0:
            result[k] = v
        else:
            raise ValueError("malformed fit field")
    return result


def _checkpoint_digest(fit: dict[str, Any]) -> str:
    value = fit.get("checkpoint_sha256", fit.get("checkpoint"))
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError("missing checkpoint SHA-256")
    if "checkpoint" in fit and "checkpoint_sha256" in fit and fit["checkpoint"] != fit["checkpoint_sha256"]:
        raise ValueError("inconsistent checkpoint digest")
    return value


def _policy_rows(src: dict[str, Any], m: dict[str, Any], arch: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    conditions = src.get("policy_aliases", {}).get(arch, [arch])
    sr = m.get("seed_rates", {}).get(str(seed), {})
    sw = m.get("seed_penalized_work", {}).get(str(seed), {})
    for condition in conditions:
        if condition in sr:
            row = {"study_id": src["study_id"], "policy": condition, "scope": "per_training_seed",
                   "split": "calibration", "independent_roots": src["independent_roots"],
                   "budget": src["reference_budget"], "closure_rate": sr[condition]}
            if condition in sw:
                row["mean_penalized_work"] = sw[condition]
            rows.append(row)
    return rows


def create_model(src: dict[str, Any], m: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    arch = _name(fit.get("arch", fit.get("arm")))
    seed = fit["seed"]
    params = fit["parameters"]
    if type(seed) is not int or seed < 0 or type(params) is not int or params < 1:
        raise ValueError("invalid model fit")
    passes = fit.get("passes", src["registered_recipe"].get("registered_passes"))
    if passes is not None and (type(passes) is not int or passes < 1):
        raise ValueError("invalid recurrent pass count")
    digest = _checkpoint_digest(fit)
    study = src["study_id"]
    model_id = f"sha256:{digest}"
    provenance = {
        "repository": src["repository"], "commit": src["commit"],
        "manifest_uri": uri(src, src["manifest_path"]), "manifest_git_blob_sha1": src["manifest_git_blob_sha1"],
        "registration_uri": uri(src, src["registration_path"]),
        "report_uri": uri(src, src["report_path"]), "evidence": "source_reported",
    }
    folder = str(PurePosixPath(src["manifest_path"]).parent / f"{seed}_{arch}")
    return {
        "schema_version": SCHEMA, "record_type": "model", "model_id": model_id,
        "notation": f"MA1[{src['family']}/{study}/{arch}@s{seed};P={params};K={passes}]",
        "family": src["family"], "population_group": src["family"], "study_id": study,
        "genotype": {"architecture": arch, "parameter_count": params, "passes": passes,
                     "implementation_uri": uri(src, src["implementation_path"])},
        "training_recipe": {"declared": src["registered_recipe"], "training_seed": seed,
                            "reported_fit_receipt": _fit_receipt(fit)},
        "phenotype": _policy_rows(src, m, arch, seed),
        "mechanistic_profile": {"study_ref": study, "scope": "study_aggregate_not_seed_specific"},
        "causal_evidence": {"study_ref": study, "scope": "study_aggregate_not_seed_specific"},
        "formal_properties": {"study_ref": study, "scope": "reported_theorems_and_certificate_receipts"},
        "lineage": {"architecture_predecessor_study": src.get("architecture_predecessor"),
                    "weight_parent": None, "group_id": src["family"]},
        "checkpoint": {"reported_sha256": digest, "verified_bytes": False,
                       "archive_directory_uri": uri(src, folder).replace("/blob/", "/tree/")},
        "provenance": provenance, "training_ready": False, "data_rights_review": "pending",
        "data_split": "unassigned", "custody_lane": "discovery",
    }


def create_study(src: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    fields = PHASE_FIELDS if src["adapter"] == "lbf18_calibration" else STUDY_FIELDS
    stats = {k: _summary(m[k]) for k in fields if k in m}
    if src["adapter"] == "lbf18_calibration":
        for key in ("clean_inference_benchmark", "symbolic_controls_benchmark"):
            stats[key] = _summary(m.get(key, {}))
        audit = m.get("soundness_and_guard_audit", {})
        stats["guard_audit_nested_pairs"] = _summary(audit.get("nested_pair_enriched_100_cohort", {}))
        # Explicit field allowlist; no raw fixture words or root identities.
        stats["float32_margin_sweep"] = [
            {**_summary(r), "architecture": _name(r["arch"])}
            for r in audit.get("ce3_float32_depth_margin_sweep", [])
        ]
    return {
        "schema_version": SCHEMA, "record_type": "study", "study_id": src["study_id"],
        "family": src["family"], "population_group": src["family"],
        "split": "calibration", "independent_roots": src["independent_roots"],
        "reference_budget": src["reference_budget"], "metrics_scope": "aggregate_not_independent_seed_episodes",
        "stats": stats, "interpretation_notes": src.get("interpretation_notes", []),
        "mechanistic_artifact_uris": [uri(src, p) for p in src.get("mechanistic_artifact_paths", [])],
        "provenance": {"manifest_uri": uri(src, src["manifest_path"]), "commit": src["commit"],
                       "manifest_git_blob_sha1": src["manifest_git_blob_sha1"],
                       "report_uri": uri(src, src["report_path"]), "evidence": "source_reported"},
        "training_ready": False, "data_rights_review": "pending",
        "data_split": "unassigned", "custody_lane": "discovery",
    }


def collate(sources: list[dict[str, Any]], cache: Path) -> tuple[list[dict], list[dict]]:
    models: dict[str, dict] = {}
    by_fit: dict[tuple[str, str, int], dict] = {}
    studies = []
    later = []
    for src in sources:
        m = verify_manifest(src, _read_bounded(cache / f"{src['source_id']}.json"))
        studies.append(create_study(src, m))
        phase = src["adapter"] == "lbf18_calibration"
        fits = m["hybrid_fits" if phase else "fits"]
        if not isinstance(fits, list) or not fits:
            raise ValueError("empty fit roster")
        for fit in fits:
            record = create_model(src, m, fit)
            key = record["model_id"]
            if key in models:
                raise ValueError(f"duplicate checkpoint fit {key}")
            identity = (src["study_id"], record["genotype"]["architecture"], fit["seed"])
            if identity in by_fit:
                raise ValueError("duplicate study/architecture/seed identity")
            models[key] = record
            by_fit[identity] = record
        if phase:
            later.append((src, m))
    for src, m in later:
        for policy, arch in src["reuse_policy_map"].items():
            b = m["clean_inference_benchmark"][policy]
            for seed_text, closure in b["per_seed_closure_b128"].items():
                seed = int(seed_text)
                fit = by_fit.get((src["study_id"], arch, seed))
                if fit is None:
                    fit = by_fit[(src["reuses_study"], arch, seed)]
                fit["phenotype"].append({
                    "study_id": src["study_id"], "policy": policy, "scope": "per_training_seed",
                    "split": "calibration", "independent_roots": b["independent_roots"],
                    "budget": 128, "closure_rate": closure,
                    "mean_penalized_work": b["per_seed_penalized_work_b128"][seed_text],
                })
    return sorted(models.values(), key=lambda x: x["notation"]), sorted(studies, key=lambda x: x["study_id"])


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = b"".join(canonical(row) + b"\n" for row in rows)
    path.write_bytes(data)
    return sha256(data)


def emit(out: Path, models: list[dict], studies: list[dict]) -> dict[str, Any]:
    # Exclusive output directory; a second run can reproduce bytes in a new directory.
    out.mkdir(parents=True, exist_ok=False)
    documents = []
    for row in [*models, *studies]:
        key = row.get("model_id", row["study_id"])
        # This is a research-corpus candidate, not a released SFT or LM example.
        documents.append({
            "schema_version": SCHEMA, "document_id": key,
            "population_group": row["population_group"], "custody_lane": "discovery",
            "training_ready": False, "data_rights_review": "pending",
            "data_split": "unassigned", "text": canonical(row).decode("utf-8"),
        })
    hashes = {"models.jsonl": write_jsonl(out / "models.jsonl", models),
              "studies.jsonl": write_jsonl(out / "studies.jsonl", studies),
              "candidate_documents.jsonl": write_jsonl(out / "candidate_documents.jsonl", documents)}
    inventory = {
        "schema_version": SCHEMA, "model_count": len(models), "study_count": len(studies),
        "candidate_document_count": len(documents), "training_ready": False,
        "training_examples_released": 0, "weights_loaded": 0,
        "model_count_by_study": dict(Counter(m["study_id"] for m in models)),
        "independent_population_groups": sorted({m["population_group"] for m in models}),
        "phenotype_rows": sum(len(m["phenotype"]) for m in models),
        "output_sha256": hashes, "rights_and_lineage_review_required": True,
    }
    (out / "inventory.json").write_bytes(canonical(inventory) + b"\n")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fetch-metadata", action="store_true",
                        help="Explicitly fetch only pinned public JSON metadata; default is offline.")
    args = parser.parse_args()
    sources = load_sources(args.sources)
    if args.fetch_metadata:
        fetch_sources(sources, args.cache)
    models, studies = collate(sources, args.cache)
    print(json.dumps(emit(args.out, models, studies), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
