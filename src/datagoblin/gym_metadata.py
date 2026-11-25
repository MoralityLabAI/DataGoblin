"""Emit a separate, offline discovery bundle for reviewed gym metadata."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .model_atlas import HEX40, HEX64, MAX_BYTES, REPO, _name, _path, canonical, write_jsonl

KINDS = {"trained_gym", "checkpoint_reuse", "study_only", "synthetic_gym", "registration_only"}
CUSTODY = {"training_ready": False, "custody_lane": "discovery", "data_split": "unassigned",
           "data_rights_review": "pending"}
PROTECTED = re.compile(r"sealed|holdout|private|raw|ordinary|ood|test_split", re.IGNORECASE)
FIELDS = {"id", "family", "kind", "scope", "sources", "architecture", "recipe", "metrics",
          "checkpoints", "limits"}
CHECKPOINT_FIELDS = {"sha256", "training_seed", "steps", "calibration_accuracy",
                     "post_sequence_passes"}


def collate(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with path.open("rb") as handle:
        payload = handle.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("catalog exceeds byte limit")
    catalog = json.loads(payload)
    if catalog.get("schema_version") != "1.0" or not isinstance(catalog.get("modules"), list):
        raise ValueError("unknown gym metadata catalog")
    modules = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for module in catalog["modules"]:
        if set(module) != FIELDS or module["kind"] not in KINDS:
            raise ValueError("unexpected gym record")
        mid = _name(module["id"])
        family = _name(module["family"])
        if mid in modules or not module["sources"]:
            raise ValueError("module requires unique ID and pinned provenance")
        provenance = []
        for source in module["sources"]:
            if set(source) != {"repository", "commit", "git_blob_sha1", "path"}:
                raise ValueError("invalid provenance keys")
            repo, sha, blob = (source[k] for k in ("repository", "commit", "git_blob_sha1"))
            p = _path(source["path"])
            if (not REPO.fullmatch(repo) or not HEX40.fullmatch(sha) or not HEX40.fullmatch(blob)
                    or Path(p).suffix not in {".json", ".md"} or PROTECTED.search(p)):
                raise ValueError("source must be a pinned open metadata path")
            provenance.append({**source, "uri": f"https://github.com/{repo}/blob/{sha}/{p}"})
        has_fits = module["kind"] in {"trained_gym", "checkpoint_reuse"}
        if bool(module["checkpoints"]) != has_fits:
            raise ValueError("checkpoint references must match the module kind")
        seen = set()
        for item in module["checkpoints"]:
            digest = item["sha256"]
            if not set(item) <= CHECKPOINT_FIELDS or not HEX64.fullmatch(digest) or digest in seen:
                raise ValueError("malformed or duplicate checkpoint reference")
            seen.add(digest)
            row = checkpoints.setdefault(
                digest, {"model_id": f"sha256:{digest}", "reported_sha256": digest,
                         "verified_bytes": False, "references": [], **CUSTODY},
            )
            row["references"].append(
                {"module_id": mid, "family": family, "kind": module["kind"],
                 **{k: v for k, v in item.items() if k != "sha256"}},
            )
        modules[mid] = {**module, "sources": provenance, "population_group": family,
                        "record_type": "gym_module", **CUSTODY}
    for row in checkpoints.values():
        row["references"].sort(key=lambda ref: ref["module_id"])
    return [modules[k] for k in sorted(modules)], [checkpoints[k] for k in sorted(checkpoints)]


def emit(sources: Path, out: Path) -> dict[str, Any]:
    models, index = collate(sources)
    out.mkdir(parents=True, exist_ok=False)
    inventory = {
        "schema_version": "1.0", "module_count": len(models), "checkpoint_digest_count": len(index),
        "terminal_fit_digest_count": sum(
            any(ref["kind"] == "trained_gym" for ref in row["references"]) for row in index
        ),
        "training_examples_released": 0, **CUSTODY,
        "sha256": {
            "gym_modules.jsonl": write_jsonl(out / "gym_modules.jsonl", models),
            "checkpoint_index.jsonl": write_jsonl(out / "checkpoint_index.jsonl", index),
        },
    }
    (out / "inventory.json").write_bytes(canonical(inventory) + b"\n")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(emit(args.sources, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
