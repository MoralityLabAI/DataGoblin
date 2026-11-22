#!/usr/bin/env python3
"""Bounded, opt-in MeTTa discovery and verified upstream collection; dry-run by default."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from corpus import LICENSE_ALLOWLIST, ROOT, embedded_assertions, git_blob_sha1, imports_present, safe_file

DEFAULT_EXCLUSIONS = re.compile(
    r"(^|/)(?:corpus|tmpdata|generated|snapshots|vendor|node_modules|large_data|checkpoints)(/|$)"
    r"|(?:\b|_)(?:private_key|secret_key|password_dump)(?:\b|_)",
    re.IGNORECASE,
)
SOURCE_RUNTIME = {
    "trueagi-io/hyperon-experimental": "hyperon",
    "trueagi-io/metta-examples": "hyperon",
    "trueagi-io/PeTTa": "petta",
    "singnet/ai-dsl": "hyperon",
    "iCog-Labs-Dev/MetaMo": "hyperon",
    "MesTTo/MeTTa-Library-Pack": "upstream-specific",
    "F1R3FLY-io/MeTTa-Compiler": "compiler",
    "singnet/Omega": "omega",
    "zariuq/CeTTa": "cetta",
    "trueagi-io/PLN": "hyperon",
    "MesTTo/MeTTaScript": "mettascript",
    "singnet/das": "das",
}


def family_for_path(path: str) -> str:
    value = path.lower()
    for needle, family in (
        ("json", "json_curation"),
        ("csv", "table_curation"),
        ("ontology", "knowledge_graph_curation"),
        ("knowledge", "knowledge_graph_curation"),
        ("graph", "knowledge_graph_curation"),
        ("unif", "symbolic_unification"),
        ("pattern", "graph_query"),
        ("query", "graph_query"),
        ("chaining", "backward_chaining"),
        ("pln", "probabilistic_inference"),
        ("state", "mutable_state"),
        ("memory", "agent_memory"),
        ("type", "type_inference"),
        ("test", "assertion_semantics"),
    ):
        if needle in value:
            return family
    return "general_metta_source"


def kind_for_path(path: str) -> str:
    lower = path.lower()
    if Path(path).name.lower().startswith("fail_"):
        return "negative_test"
    if "/tests/" in f"/{lower}" or Path(path).name.lower().startswith("test_"):
        return "upstream_test"
    if "/bench" in f"/{lower}":
        return "benchmark"
    if "/examples/" in f"/{lower}":
        return "example"
    return "library"


def public_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "DataGoblin-Metta-Curation"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def existing_snapshot_ids(root: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    blob_shas: set[str] = set()
    for p in (root / "annotations").rglob("*.json"):
        item = json.loads(p.read_text(encoding="utf-8"))
        ids.add(item["id"])
        blob_shas.add(item["upstream_blob_sha1"])
    return ids, blob_shas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="An approved owner/repo listed in discovery_catalog.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--max-file-bytes", type=int, default=32_768)
    parser.add_argument("--max-total-bytes", type=int, default=500_000)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--include-data", action="store_true", help="Include small data/ assets after source review")
    parser.add_argument("--apply", action="store_true", help="Fetch and write, instead of the default dry run")
    args = parser.parse_args()
    root = args.root.resolve()
    catalog = json.loads((root / "discovery_catalog.json").read_text(encoding="utf-8"))
    permitted = {item["repo"]: item["license"] for item in catalog["seeded_repositories"]}
    license_id = permitted.get(args.repo)
    if license_id not in LICENSE_ALLOWLIST:
        parser.error("Repository must have an approved license and an entry in seeded_repositories")
    slug = args.repo.replace("/", "--")
    if not (root / "licenses" / slug / "LICENSE").is_file():
        parser.error("The upstream LICENSE must be saved before collecting sources")
    if not re.fullmatch(r"[^/]+/[^/]+", args.repo):
        parser.error("Invalid repository selector")
    if any(number <= 0 for number in (args.max_file_bytes, args.max_total_bytes, args.max_files)):
        parser.error("Budget values must be positive")
    pinned: set[str] = set()
    for annotation in (root / "annotations" / slug).rglob("*.json"):
        pinned.add(json.loads(annotation.read_text(encoding="utf-8"))["source_commit"])
    if len(pinned) != 1:
        parser.error("Expected exactly one pinned source revision from the seed annotation set")
    commit = next(iter(pinned))
    owner, name = args.repo.split("/")
    api = f"https://api.github.com/repos/{quote(owner)}/{quote(name)}/git/trees/{commit}?recursive=1"
    tree = public_json(api)
    if tree.get("truncated") or not isinstance(tree.get("tree"), list):
        parser.error("Upstream tree is truncated or missing; refusing an incomplete crawl")
    candidates = []
    for item in tree["tree"]:
        path = item.get("path", "")
        size = item.get("size") or 0
        if item.get("type") != "blob" or not path.endswith(".metta"):
            continue
        if args.path_prefix and not path.startswith(args.path_prefix):
            continue
        if DEFAULT_EXCLUSIONS.search(path):
            continue
        if not args.include_data and re.search(r"(^|/)(?:data|assets)(/|$)", path, re.IGNORECASE):
            continue
        if 0 < size <= args.max_file_bytes and re.fullmatch(r"[0-9a-f]{40}", item.get("sha", "")):
            candidates.append((path, size, item["sha"]))
    candidates.sort(key=lambda item: item[0])
    seen, blob_digests = existing_snapshot_ids(root)
    planned: list[tuple[str, int, str]] = []
    total = 0
    for path, size, sha in candidates:
        if f"{slug}:{path}" in seen or sha in blob_digests:
            continue
        if len(planned) >= args.max_files or total + size > args.max_total_bytes:
            continue
        planned.append((path, size, sha))
        total += size
    if not args.apply:
        print(json.dumps({
            "mode": "dry_run", "repo": args.repo, "source_commit": commit,
            "candidate_files": len(candidates), "selected_files": len(planned),
            "selected_bytes": total, "selected_paths": [path for path, _, _ in planned],
        }, indent=2))
        return 0
    written = 0
    for path, size, sha in planned:
        url = "https://raw.githubusercontent.com/" + "/".join(
            quote(part, safe="") for part in [owner, name, commit, *path.split("/")]
        )
        with urlopen(Request(url, headers={"User-Agent": "DataGoblin-Metta-Curation"}), timeout=30) as response:
            data = response.read(args.max_file_bytes + 1)
        if len(data) != size or git_blob_sha1(data) != sha:
            raise ValueError(f"Upstream Git blob digest or size mismatch: {path}")
        content = data.decode("utf-8")
        raw_path = f"raw/{slug}/{path}"
        sidecar_path = f"annotations/{slug}/{path}.json"
        role = kind_for_path(path)
        record = {
            "schema_version": "1.0",
            "id": f"{slug}:{path}",
            "raw_path": raw_path,
            "source_repo": args.repo,
            "source_commit": commit,
            "source_path": path,
            "source_url": f"https://github.com/{args.repo}/blob/{commit}/{path}",
            "upstream_blob_sha1": sha,
            "license_spdx": license_id,
            "license_evidence": "upstream repository SPDX metadata",
            "source_kind": role,
            "task_family": family_for_path(path),
            "runtime_hint": SOURCE_RUNTIME.get(args.repo, "upstream-specific"),
            "embedded_test_assertion_count": embedded_assertions(content),
            "imports_detected": imports_present(content),
            "provenance_status": "pinned_upstream_snapshot",
            "verification_status": "not_run",
            "verification_receipt": None,
            "rollout_status": "none",
            "training_status": "source_only_pending_review",
            "split": "unassigned",
            "annotations_method": "automated_path_and_source_text_heuristics",
            "utf8_bytes": len(data),
            "line_count": len(content.split("\n")),
            "model_token_count": None,
            "upstream_expected_outcome": "upstream_expected_rejection" if role == "negative_test" else "unspecified",
        }
        destination = safe_file(root, raw_path)
        sidecar = safe_file(root, sidecar_path)
        if destination.exists() or sidecar.exists():
            raise ValueError(f"Unexpected overwrite attempt: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        sidecar.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    print(f"Collected {written} source-only files from {args.repo}; verify before committing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
