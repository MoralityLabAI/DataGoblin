"""Local, hash-pinned intake of private audited-tag policy training data.

This module never downloads sources or executes code from the source repository.
Only original training rows enter its output. Keep the output in private storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

SOURCE_ID = "jinn-beast-audited-tags-v1"
REGISTRY_SCHEMA = "research-v2-held-roots-v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FORM_RE = re.compile(r"^\(([^()\s]+)(?:\s+([^()]*))?\)$")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def semantic_digest(tags: list[str]) -> str:
    return digest(canonical(sorted(tags)).encode())


def root_id(tags: list[str]) -> str:
    return "jinn-beast:" + semantic_digest(tags)


def _load_manifest(path: Path | None) -> dict[str, Any]:
    location = path or Path(__file__).resolve().parents[3] / "configs/research_v2/specialist_sources.json"
    manifest = json.loads(location.read_text(encoding="utf-8"))
    if manifest.get("schema") != "research-v2-specialist-sources-v1" or manifest.get("source_id") != SOURCE_ID:
        raise ValueError("Unsupported specialist source manifest")
    if manifest.get("visibility") != "private" or manifest.get("use") != "internal_specialist_training":
        raise ValueError("Source manifest must preserve private internal use")
    return manifest


def _read_pinned(root: Path, spec: dict[str, Any]) -> bytes:
    relative = Path(spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid source path")
    path = root / relative
    if not path.is_file():
        path = root / relative.name
    # A flat operator export and an original checkout are both supported.
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the supplied source root")
    data = path.read_bytes()
    if len(data) != spec["bytes"] or digest(data) != spec["sha256"]:
        raise ValueError(f"Pinned source mismatch: {relative.name}")
    return data


def _policy(data: bytes) -> dict[str, Any]:
    """Parse the bounded fact grammar independently of the upstream Python scorer."""
    result: dict[str, Any] = {"dimensions": {}, "tags": {}, "critical_tags": []}
    single = {"benchmark": "benchmark", "constitution": "constitution", "critical-cap": "critical_cap"}
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        match = FORM_RE.fullmatch(line)
        if not match:
            raise ValueError("Invalid policy fact")
        name, fields = match.group(1), (match.group(2) or "").split()
        if name in single:
            key = single[name]
            if len(fields) != 1 or key in result:
                raise ValueError("Duplicate or malformed policy identifier")
            result[key] = fields[0]
        elif name == "dimension":
            if len(fields) != 2 or fields[0] in result["dimensions"]:
                raise ValueError("Duplicate or malformed dimension")
            result["dimensions"][fields[0]] = fields[1]
        elif name == "tag-score":
            if len(fields) != 3 or fields[0] in result["tags"]:
                raise ValueError("Duplicate or malformed tag")
            result["tags"][fields[0]] = {"dimension": fields[1], "value": fields[2]}
        elif name == "critical-tag":
            if len(fields) != 1 or fields[0] in result["critical_tags"]:
                raise ValueError("Duplicate or malformed critical tag")
            result["critical_tags"].append(fields[0])
        elif name != "ablation" or len(fields) < 2:
            raise ValueError("Unsupported policy fact")
    if not all(key in result for key in single.values()):
        raise ValueError("Incomplete policy")
    dimensions = {name: Fraction(value) for name, value in result["dimensions"].items()}
    if not dimensions or sum(dimensions.values()) != 1 or any(value <= 0 for value in dimensions.values()):
        raise ValueError("Invalid policy weights")
    if not 0 <= Fraction(result["critical_cap"]) <= 1:
        raise ValueError("Invalid critical cap")
    for tag in result["tags"].values():
        if tag["dimension"] not in dimensions or Fraction(tag["value"]) not in {-1, 1}:
            raise ValueError("Invalid tag dimension or sign")
    if not set(result["critical_tags"]) <= set(result["tags"]):
        raise ValueError("Unknown critical tag")
    return result


def _score(tags: list[str], policy: dict[str, Any], low: Fraction, high: Fraction) -> dict[str, Any]:
    active = sorted(set(tags) & set(policy["tags"]))
    by_dimension: dict[str, set[Fraction]] = {name: set() for name in policy["dimensions"]}
    for tag in active:
        spec = policy["tags"][tag]
        by_dimension[spec["dimension"]].add(Fraction(spec["value"]))
    conflicts = sorted(name for name, signs in by_dimension.items() if len(signs) > 1)
    critical = sorted(set(active) & set(policy["critical_tags"]))
    if not active or conflicts:
        score, lane = Fraction(1, 2), "hold"
    else:
        total = sum(
            Fraction(policy["dimensions"][name]) * sum(signs)
            for name, signs in by_dimension.items()
        )
        score = (total + 1) / 2
        if critical:
            score, lane = min(score, Fraction(policy["critical_cap"])), "reject"
        else:
            lane = "train" if score >= high else "reject" if score <= low else "hold"
    return {
        "lane": lane,
        "score": round(float(score), 6),
        "critical_hits": critical,
        "annotation_conflicts": conflicts,
        "relevant_tags": active,
    }


def _registry(path: Path) -> tuple[set[str], set[str], str]:
    data = path.read_bytes()
    value = json.loads(data)
    if value.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("A prospective held-root registry is required")
    for key in ("held_root_ids", "held_semantic_sha256", "allow_source_ids"):
        if not isinstance(value.get(key), list) or any(not isinstance(item, str) for item in value[key]):
            raise ValueError(f"Registry requires a string list: {key}")
    if SOURCE_ID not in value["allow_source_ids"]:
        raise ValueError("Registry does not allow this source for internal training")
    if any(not SHA_RE.fullmatch(value) for value in value["held_semantic_sha256"]):
        raise ValueError("Invalid held semantic digest")
    return set(value["held_root_ids"]), set(value["held_semantic_sha256"]), digest(data)


def import_specialist(
    source_root: Path,
    output_dir: Path,
    held_registry: Path,
    *,
    manifest_path: Path | None = None,
    augment: bool = False,
) -> dict[str, Any]:
    """Verify source bytes/labels, then atomically stage private training-only rows.

    The registry permits the named source and binds prospective held families by
    root ID and semantic tag-set digest. Original validation/test rows always add
    their digests to that exclusion set. Counterfactual views retain their parent
    root and are omitted if they collide with any held tag set.
    """
    source_root, output_dir, held_registry = map(Path, (source_root, output_dir, held_registry))
    if output_dir.exists():
        raise ValueError("Output must be a fresh directory")
    manifest = _load_manifest(manifest_path)
    held_roots, held_semantics, registry_sha = _registry(held_registry)
    payloads = {name: _read_pinned(source_root, spec) for name, spec in manifest["files"].items()}
    policies = {frame: _policy(payloads[frame]) for frame in ("jinn", "beast")}
    low, high = Fraction(manifest["reject_maximum"]), Fraction(manifest["train_minimum"])
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid lane thresholds")
    universe = set().union(*(set(policy["tags"]) for policy in policies.values()))
    rows = [json.loads(line) for line in payloads["dataset"].decode("utf-8").splitlines() if line]
    seen: set[str] = set()
    seen_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("schema_version") != manifest["source_row_schema"]:
            raise ValueError("Unexpected source row schema")
        tags = row.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("Audited tags must be a list of strings")
        if len(tags) != len(set(tags)) or not set(tags) <= universe:
            raise ValueError("Duplicate or unknown audited tag")
        semantic = semantic_digest(tags)
        if semantic in seen or row.get("row_id") in seen_ids:
            raise ValueError("Duplicate tag family or row identifier across source splits")
        if row.get("row_id") != "mb-" + semantic[:16]:
            raise ValueError("Source row identifier does not bind its tag set")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Unknown source split")
        seen.add(semantic)
        seen_ids.add(row["row_id"])
        split_counts[row["split"]] += 1
        for frame, policy in policies.items():
            actual = _score(tags, policy, low, high)
            label = row.get("labels", {}).get(frame, {})
            if any(label.get(key) != value for key, value in actual.items()):
                raise ValueError(f"Independent policy check failed for {row['row_id']} / {frame}")
            if row.get("joint_label", {}).get(frame) != actual["lane"]:
                raise ValueError("Joint label differs from policy result")
        if row["split"] != "train":
            held_roots.add(root_id(tags))
            held_semantics.add(semantic)
    if len(rows) != manifest["expected_rows"] or dict(split_counts) != manifest["expected_splits"]:
        raise ValueError("Source split counts differ from the pinned manifest")
    output: list[dict[str, Any]] = []
    admitted_roots: set[str] = set()
    held_collisions = Counter()
    emitted: set[tuple[str, str]] = set()

    def emit(row: dict[str, Any], frame: str, tags: list[str], variant: str) -> None:
        root, semantic = root_id(row["tags"]), semantic_digest(tags)
        if root in held_roots or semantic in held_semantics:
            held_collisions[variant] += 1
            return
        key = (frame, semantic)
        if key in emitted:
            held_collisions["duplicate_view"] += 1
            return
        emitted.add(key)
        policy = policies[frame]
        target = _score(tags, policy, low, high)
        output.append({
            "schema": "research-v2-specialist-record-v1",
            "record_id": f"{root}:{frame}:{variant}",
            "root_id": root,
            "semantic_sha256": semantic,
            "original_row_id": row["row_id"],
            "split": "train",
            "source_id": SOURCE_ID,
            "source_revision": manifest["revision"],
            "source_sha256": manifest["files"]["dataset"]["sha256"],
            "policy_sha256": manifest["files"][frame]["sha256"],
            "policy_frame": frame,
            "variant": variant,
            "visibility": "private",
            "use": "internal_specialist_training",
            "claim_scope": manifest["claim_scope"],
            "input": {"task": "classify_audited_tags_under_supplied_policy", "tags": sorted(tags),
                      "policy": policy, "train_minimum": str(high), "reject_maximum": str(low)},
            "target": target,
        })
        admitted_roots.add(root)

    # Emit originals first so optional views never displace original training rows.
    for row in rows:
        if row["split"] == "train":
            for frame in policies:
                emit(row, frame, row["tags"], "original")
    if augment:
        for row in rows:
            if row["split"] == "train":
                for frame, policy in policies.items():
                    own = sorted(set(row["tags"]) & set(policy["tags"]))
                    if own != sorted(row["tags"]):
                        emit(row, frame, own, "own_frame_view")
    output.sort(key=lambda row: row["record_id"])
    held = {"schema": REGISTRY_SCHEMA, "held_root_ids": sorted(held_roots),
            "held_semantic_sha256": sorted(held_semantics), "allow_source_ids": [SOURCE_ID]}
    files = {
        "train.jsonl": "".join(canonical(row) + "\n" for row in output).encode(),
        "held_roots.json": (canonical(held) + "\n").encode(),
    }
    receipt = {
        "schema": "research-v2-specialist-receipt-v1",
        "source_id": SOURCE_ID,
        "source_revision": manifest["revision"],
        "source_manifest_sha256": digest(canonical(manifest).encode()),
        "source_files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in payloads.items()},
        "held_registry_sha256": registry_sha,
        "original_splits": dict(sorted(split_counts.items())),
        "original_train_roots": split_counts["train"],
        "admitted_train_roots": len(admitted_roots),
        "training_targets": len(output),
        "targets_by_frame": dict(sorted(Counter(row["policy_frame"] for row in output).items())),
        "targets_by_variant": dict(sorted(Counter(row["variant"] for row in output).items())),
        "excluded_views": dict(sorted(held_collisions.items())),
        "held_semantic_count": len(held_semantics),
        "all_source_labels_recomputed": True,
        "train_only": True,
        "visibility": "private",
        "use": "internal_specialist_training",
        "public_redistribution": manifest["public_redistribution"],
        "claims": manifest["claim_scope"],
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in files.items()},
    }
    files["RECEIPT.json"] = (canonical(receipt) + "\n").encode()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".specialist-", dir=output_dir.parent))
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        if output_dir.exists():
            raise ValueError("Output appeared while the import was running")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-registry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args(argv)
    receipt = import_specialist(args.source_root, args.out, args.held_registry,
                                manifest_path=args.manifest, augment=args.augment)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
