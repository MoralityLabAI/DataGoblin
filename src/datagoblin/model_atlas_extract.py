"""Metadata-only extraction of checkpoint receipts from a pinned Git commit.

Reads Git objects through ``git cat-file``; never checks out a worktree, never reads weight
blobs (only their tree entries: path, size and Git blob SHA-1), never executes upstream code and
never copies strings other than short allowlisted identifiers. Output is deterministic for a
given commit and ``--prefix`` selection. Run ``python -m datagoblin.model_atlas_extract --help``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from datagoblin.model_atlas import HEX40, HEX64, MAX_BYTES, REPO, canonical, sha256

EXTRACT_SCHEMA = "atlas-extract/1"
CONTAINER_DIRS = frozenset({"experiments", "research", "mechinterp", "packages", "studies", "results", "runs"})
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".ckpt", ".npz", ".bin", ".gguf", ".onnx", ".h5")
DOC_NAMES = re.compile(
    r"(readme|report|result|registration|preregist|protocol|analysis|referee|errata|correction|invalid|negative|"
    r"summary|findings|notes?|specification|audit)", re.I)
# Evaluation custody: sealed cohorts, private witnesses, hidden seeds, labels. Tokens, not substrings.
CUSTODY_TOKENS = frozenset({
    "sealed", "holdout", "hold", "hidden", "private", "ordinary", "ood", "witness", "witnesses", "secret",
    "secrets", "labels", "answer", "answers", "heldout", "confirmation_sealed", "credentials", "token", "tokens",
})
CUSTODY_KEY_WORDS = (
    "witness", "secret", "token", "label", "generator_seed", "sealed", "root_ids", "lineage_ids", "answer",
    "hidden", "holdout", "private", "password",
)
NON_VALUE_KEY_WORDS = ("sha", "hash", "path", "file", "uri", "url", "commit", "utc", "timestamp", "_at", "digest")
CKPT_KEY = re.compile(r"(checkpoint|ckpt|model|weights?|state_dict)")
CKPT_KEY_SUFFIX = re.compile(r"(sha256|_sha|_hash|_digest)")
CKPT_KEY_REJECT = re.compile(r"(source_sha|_src|code_|config|manifest|optimizer|decomposition|probe|calibration|"
                             r"dataset|sample|"
                             r"tokenizer|answer_tape|draws|trace|environment|git_blob|zip|artifact_|archive|lock)")
LABEL_KEYS = frozenset({
    "arch", "arm", "architecture", "model", "model_name", "name", "variant", "kind", "topology", "mode", "family",
    "optimizer", "dtype", "precision", "quantization", "schedule", "lr_schedule", "device", "type",
})
SHORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+\-]{0,47}\Z")
KEY_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}\Z")
VERDICT_KEYS = re.compile(r"(passed|failed|verdict|supported|refuted|falsified|status|success|confirmed|claim_|"
                          r"valid|negative|invalid)", re.I)
MAX_FIELDS = 64
MAX_DEPTH = 14
MAX_LIST = 20000
EVIDENCE_TAGS = {
    "mechanistic": ("sae", "probe", "activation", "rank", "gate", "lens", "cka", "geometry", "neuron", "feature",
                    "interp", "mechinterp", "decomp", "attribution", "trajector", "spectral", "floquet"),
    "causal": ("ablat", "patch", "intervention", "causal", "sham", "lesion", "counterfactual", "control", "swap",
               "paired"),
    "formal": ("lean", "theorem", "proof", "certificate", "verified", "formal", "invariant", "equivarian",
               "counterexample", "smt", "coq"),
}


class ObjectSource(Protocol):
    """Read-only view of one immutable commit."""

    commit: str

    def tree(self) -> list[tuple[str, str, int]]: ...

    def blobs(self, shas: list[str]) -> dict[str, bytes]: ...


class GitSource:
    """Read objects of one commit with the Git CLI; no checkout, no hooks, no filters."""

    sizes = True

    def __init__(self, git_dir: Path, ref: str, git: str | None = None):
        self.git_dir = git_dir
        self.git = git or os.environ.get("GIT", "git")
        self.commit = self._run("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        if not HEX40.fullmatch(self.commit):
            raise ValueError("ref did not resolve to a commit")

    def _run(self, *args: str, stdin: bytes | None = None) -> bytes:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "GIT_NO_LAZY_FETCH": "0"}
        done = subprocess.run([self.git, "-C", str(self.git_dir), "--no-pager", *args], input=stdin,
                              capture_output=True, check=True, env=env)
        return done.stdout

    def tree(self) -> list[tuple[str, str, int]]:
        flags = ["-l"] if self.sizes else []
        out = self._run("ls-tree", "-r", *flags, "-z", self.commit).decode("utf-8", "surrogateescape")
        rows = []
        for entry in out.split("\0"):
            if not entry:
                continue
            meta, path = entry.split("\t", 1)
            fields = meta.split()
            mode, kind, sha = fields[:3]
            size = fields[3] if self.sizes else "-"
            if kind == "blob" and mode != "120000":
                rows.append((path, sha, -1 if size == "-" else int(size)))
        return sorted(rows)

    def blobs(self, shas: list[str]) -> dict[str, bytes]:
        if not shas:
            return {}
        data = self._run("cat-file", "--batch", stdin="".join(f"{s}\n" for s in shas).encode())
        out: dict[str, bytes] = {}
        pos = 0
        for sha in shas:
            end = data.index(b"\n", pos)
            header = data[pos:end].split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError(f"missing blob {sha}")
            size = int(header[2])
            out[sha] = data[end + 1:end + 1 + size]
            pos = end + 1 + size + 1
        return out


class RawGitHubSource(GitSource):
    """Tree from a blobless clone; JSON bytes from the pinned-commit raw URL, verified against the tree's blob SHA-1."""

    sizes = False  # sizes would force a lazy fetch of every blob in a blobless clone

    def __init__(self, git_dir: Path, ref: str, repository: str, git: str | None = None, workers: int = 8):
        super().__init__(git_dir, ref, git)
        self.repository = repository
        self.workers = workers
        self._paths: dict[str, str] = {}

    def tree(self) -> list[tuple[str, str, int]]:
        rows = super().tree()
        self._paths = {sha: path for path, sha, _ in rows}
        return rows

    def _one(self, sha: str) -> tuple[str, bytes | None]:
        import urllib.parse
        import urllib.request

        from datagoblin.model_atlas import git_blob_sha1
        url = f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{urllib.parse.quote(self._paths[sha])}"
        request = urllib.request.Request(url, headers={"User-Agent": "DataGoblin-model-atlas/1"})
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as handle:
                    data = handle.read(MAX_BYTES + 1)
                return sha, data if git_blob_sha1(data) == sha else None
            except OSError:
                continue
        return sha, None

    def blobs(self, shas: list[str]) -> dict[str, bytes]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as pool:
            got = dict(pool.map(self._one, shas))
        return {sha: data if data is not None else b"not-json" for sha, data in got.items()}


class MemorySource:
    """Fixture source for tests; mirrors GitSource over an in-memory tree."""

    def __init__(self, commit: str, files: dict[str, bytes]):
        from datagoblin.model_atlas import git_blob_sha1
        self.commit = commit
        self._files = {path: (git_blob_sha1(data), data) for path, data in files.items()}

    def tree(self) -> list[tuple[str, str, int]]:
        return sorted((p, s, len(d)) for p, (s, d) in self._files.items())

    def blobs(self, shas: list[str]) -> dict[str, bytes]:
        by_sha = {s: d for s, d in self._files.values()}
        return {s: by_sha[s] for s in shas}


def tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def custody_reason(path: str) -> str | None:
    for part in PurePosixPath(path).parts:
        hit = tokens(part) & CUSTODY_TOKENS
        if hit:
            return f"custody_token:{sorted(hit)[0]}"
        # compound names such as sealed_confirmation or ordinary_suite
        low = part.lower()
        for word in ("sealed", "holdout", "heldout", "hidden", "private", "witness"):
            if word in low:
                return f"custody_token:{word}"
    return None


def study_of(repo_path: str) -> str:
    parts = repo_path.split("/")
    if len(parts) == 1:
        return "."
    if parts[0] in CONTAINER_DIRS and len(parts) > 2:
        if len(parts) > 4 and parts[2] == "results":
            return "/".join(parts[:4])
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def is_ckpt_key(key: str) -> bool:
    low = key.lower()
    return bool(CKPT_KEY.search(low) and CKPT_KEY_SUFFIX.search(low) and not CKPT_KEY_REJECT.search(low))


def _key_ok(key: str) -> bool:
    low = key.lower()
    return bool(KEY_OK.fullmatch(key)) and not any(w in low for w in CUSTODY_KEY_WORDS)


def _keepable(value: Any, low_key: str) -> bool:
    if type(value) is bool or type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    return isinstance(value, str) and low_key in LABEL_KEYS and bool(SHORT_ID.fullmatch(value))


def scalar_fields(node: dict[str, Any]) -> dict[str, Any]:
    """Own numeric/Boolean fields and short identifier strings; nothing else is copied."""
    out: dict[str, Any] = {}
    for key in sorted(node):
        value = node[key]
        if not isinstance(key, str) or not _key_ok(key):
            continue
        low = key.lower()
        if any(w in low for w in NON_VALUE_KEY_WORDS):
            continue
        if _keepable(value, low):
            out[key] = value
        if len(out) >= MAX_FIELDS:
            break
    return out


def verdict_fields(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(node):
        value = node[key]
        is_verdict = isinstance(key, str) and _key_ok(key) and VERDICT_KEYS.search(key)
        if is_verdict and (type(value) is bool or (isinstance(value, str) and SHORT_ID.fullmatch(value))):
            out[key] = value
    return out


def role_of(key: str) -> str:
    return key.lower()


def artifact_names(node: dict[str, Any]) -> list[str]:
    """Basenames of weight files named by the receipt; names only, never the directories."""
    names = set()
    for value in node.values():
        if isinstance(value, str) and value.lower().endswith(WEIGHT_SUFFIXES):
            base = value.replace("\\", "/").rsplit("/", 1)[-1]
            if KEY_OK.fullmatch(base):
                names.add(base)
    return sorted(names)


RECIPE_CONTAINERS = ("args", "config", "hparams", "hyperparameters", "recipe", "training", "train_args")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}\Z")
MODEL_ID_KEYS = frozenset({"base", "base_model", "teacher", "teacher_model", "init_from", "pretrained"})


def is_weight_sha_key(key: str, node: dict[str, Any]) -> bool:
    """A bare ``sha256`` beside a weight-file name identifies that file's bytes."""
    if key.lower() not in ("sha256", "sha"):
        return False
    return any(isinstance(v, str) and v.lower().endswith(WEIGHT_SUFFIXES) for v in node.values())


LOCAL_DIRS = frozenset({"checkpoints", "runs", "out", "output", "results", "models", "data", "weights", "ckpt"})


def model_ids(node: dict[str, Any]) -> dict[str, str]:
    return {k: v for k, v in sorted(node.items())
            if isinstance(v, str) and k.lower() in MODEL_ID_KEYS and MODEL_ID.fullmatch(v)
            and v.split("/", 1)[0].lower() not in LOCAL_DIRS}


def hashed_refs(node: dict[str, Any]) -> list[dict[str, str]]:
    """Checkpoint-like digests held by an enclosing dict, e.g. a teacher or base digest beside the run's files."""
    return [{"role": k.lower(), "sha256": v} for k, v in sorted(node.items())
            if isinstance(k, str) and isinstance(v, str) and HEX64.fullmatch(v)
            and (is_ckpt_key(k) or (k.lower().startswith(("teacher_", "base_")) and "sha" in k.lower()))
            and "manifest" not in k.lower() and "config" not in k.lower()]


def find_observations(obj: Any) -> list[dict[str, Any]]:
    """Dicts that carry a checkpoint digest, with their own scalars and two ancestor levels."""
    found: list[dict[str, Any]] = []

    def walk(node: Any, pointer: str, ancestors: list[dict[str, Any]], depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        if isinstance(node, dict):
            own = scalar_fields(node)
            own_ids = model_ids(node)
            ctx: dict[str, Any] = {}
            refs: list[dict[str, str]] = []
            for anc in ancestors[-2:]:
                for k, v in anc["own"].items():
                    ctx.setdefault(k, v)
                for k, v in anc["recipe"].items():
                    ctx.setdefault(k, v)
                for k, v in anc["ids"].items():
                    ctx.setdefault(k, v)
                refs.extend(anc["refs"])
            for key in sorted(node):
                value = node[key]
                if not (isinstance(key, str) and isinstance(value, str) and HEX64.fullmatch(value)):
                    continue
                if is_ckpt_key(key) or is_weight_sha_key(key, node):
                    role = "checkpoint_file_sha256" if key.lower() in ("sha256", "sha") else role_of(key)
                    found.append({"pointer": f"{pointer}/{key}", "role": role, "sha256": value,
                                  "fields": own, "context": ctx, "verdicts": verdict_fields(node),
                                  "artifact_names": artifact_names(node), "ancestor_refs": refs,
                                  "model_ids": {**{k: v for k, v in ctx.items() if k in MODEL_ID_KEYS}, **own_ids}})
            recipe: dict[str, Any] = {}
            for name in RECIPE_CONTAINERS:
                if isinstance(node.get(name), dict):
                    recipe.update(scalar_fields(node[name]))
                    own_ids = {**own_ids, **model_ids(node[name])}
            frame = {"own": own, "recipe": recipe, "ids": own_ids, "refs": hashed_refs(node)}
            for key in sorted(node, key=str):
                if isinstance(key, str):
                    walk(node[key], f"{pointer}/{key}", [*ancestors, frame], depth + 1)
        elif isinstance(node, list):
            for i, item in enumerate(node[:MAX_LIST]):
                walk(item, f"{pointer}[{i}]", ancestors, depth + 1)

    walk(obj, "", [], 0)
    return found


def loop_summaries(results: Any) -> dict[str, Any]:
    """Aggregate per-loop summaries and multiple-choice accuracies of a run's ``results_file``."""
    metrics: dict[str, Any] = {}
    denominators: dict[str, Any] = {}
    per_loop = results.get("per_loop") if isinstance(results, dict) else None
    if not isinstance(per_loop, dict):
        return {}
    for loops in sorted(per_loop, key=str):
        block = per_loop[loops]
        if not isinstance(block, dict) or not KEY_OK.fullmatch(str(loops)):
            continue
        summary = block.get("summary")
        for key, value in scalar_fields(summary if isinstance(summary, dict) else {}).items():
            metrics[f"loops{loops}.{key}"] = value
        for key in ("flops_per_token_256", "layer_evals", "latency_ms_b8_s256", "act_peak_mb_b8_s256"):
            if type(block.get(key)) in (int, float) and math.isfinite(block[key]):
                metrics[f"loops{loops}.{key}"] = block[key]
        mc = block.get("mc")
        if isinstance(mc, dict):
            for task in sorted(mc):
                row = mc[task]
                if isinstance(row, dict) and KEY_OK.fullmatch(task):
                    for key in ("acc", "acc_norm"):
                        if type(row.get(key)) in (int, float):
                            metrics[f"loops{loops}.mc.{task}.{key}"] = row[key]
                    if type(row.get("n")) is int:
                        denominators[f"loops{loops}.mc.{task}.n"] = row["n"]
    return {"metrics": dict(list(metrics.items())[:400]), "denominators": denominators}


def key_inventory(obj: Any, limit: int = 400) -> list[str]:
    names: Counter[str] = Counter()

    def walk(node: Any, depth: int) -> None:
        if depth > 8 or len(names) > limit:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and KEY_OK.fullmatch(key):
                    names[key.lower()] += 1
                walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node[:200]:
                walk(item, depth + 1)

    walk(obj, 0)
    return sorted(names)


def evidence_tags(strings: list[str]) -> list[str]:
    blob = " ".join(strings).lower()
    return sorted(tag for tag, words in EVIDENCE_TAGS.items() if any(w in blob for w in words))


def extract(source: ObjectSource, repository: str, prefixes: list[str] | None = None) -> dict[str, Any]:
    if not REPO.fullmatch(repository):
        raise ValueError("invalid repository")
    tree = [row for row in source.tree() if not prefixes or any(row[0].startswith(p) for p in prefixes)]
    excluded: Counter[str] = Counter()
    excluded_examples: dict[str, str] = {}
    studies: dict[str, dict[str, Any]] = {}

    def study(path: str) -> dict[str, Any]:
        return studies.setdefault(study_of(path), {
            "docs": [], "receipts": [], "weights_like": [], "file_count": 0, "jsonl_not_collected": 0,
            "oversize_json": 0, "unparsable_json": 0, "other_files": 0,
        })

    json_rows: list[tuple[str, str, int]] = []
    for path, sha, size in tree:
        entry = study(path)
        entry["file_count"] += 1
        reason = custody_reason(path)
        lower = path.lower()
        if lower.endswith(WEIGHT_SUFFIXES):
            if reason:
                excluded[reason] += 1
                excluded_examples.setdefault(reason, path)
            else:
                entry["weights_like"].append({"path": path, "git_blob_sha1": sha, "size": None if size < 0 else size})
        elif reason:
            excluded[reason] += 1
            excluded_examples.setdefault(reason, path)
        elif lower.endswith(".json"):
            if size > MAX_BYTES:
                entry["oversize_json"] += 1
            else:
                json_rows.append((path, sha, size))
        elif lower.endswith(".jsonl"):
            entry["jsonl_not_collected"] += 1
        elif lower.endswith(".md") and DOC_NAMES.search(PurePosixPath(path).name):
            entry["docs"].append({"path": path, "git_blob_sha1": sha})
        else:
            entry["other_files"] += 1

    observations: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    unique = sorted({sha for _, sha, _ in json_rows})
    for i in range(0, len(unique), 512):
        blobs.update(source.blobs(unique[i:i + 512]))
    by_path = {path: sha for path, sha, _ in json_rows}
    for path, sha, size in json_rows:
        entry = study(path)
        try:
            obj = json.loads(blobs[sha])
        except (ValueError, RecursionError):
            entry["unparsable_json"] += 1
            continue
        found = find_observations(obj)
        sibling: dict[str, Any] = {}
        if found and isinstance(obj, dict) and isinstance(obj.get("results_file"), str):
            target = str(PurePosixPath(path).parent / obj["results_file"])
            if target in by_path and not custody_reason(target):
                try:
                    summary = loop_summaries(json.loads(blobs[by_path[target]]))
                except (ValueError, RecursionError):
                    summary = {}
                if summary:
                    sibling = {"source_path": target, "source_git_blob_sha1": by_path[target], **summary}
        keys = key_inventory(obj) if found or any(t in path.lower() for t in ("manifest", "summary", "result")) else []
        receipt = {"path": path, "git_blob_sha1": sha, "size": None if size < 0 else size, "observations": len(found),
                   "evidence_tags": evidence_tags([path, *keys])}
        if isinstance(obj, dict):
            schema = obj.get("schema") or obj.get("schema_version")
            if isinstance(schema, str) and SHORT_ID.fullmatch(schema):
                receipt["schema"] = schema
            verdicts = verdict_fields(obj)
            if verdicts:
                receipt["top_level_verdicts"] = verdicts
        if found or receipt["evidence_tags"] or "manifest" in path.lower() or "summary" in path.lower():
            entry["receipts"].append(receipt)
        for obs in found:
            row = {"study": study_of(path), "receipt_path": path, "receipt_git_blob_sha1": sha, **obs}
            if sibling:
                row["sibling_results"] = sibling
            observations.append(row)

    for entry in studies.values():
        entry["docs"].sort(key=lambda d: d["path"])
        entry["receipts"].sort(key=lambda r: r["path"])
        entry["weights_like"].sort(key=lambda w: w["path"])
        entry["evidence_tags"] = sorted({t for r in entry["receipts"] for t in r["evidence_tags"]} |
                                        set(evidence_tags([d["path"] for d in entry["docs"]])))
        entry["negative_or_correction_docs"] = [
            d["path"] for d in entry["docs"]
            if re.search(r"(invalid|negative|errata|correction|referee|retract|refut)", d["path"], re.I)]
    observations.sort(key=lambda o: (o["receipt_path"], o["pointer"], o["role"]))
    return {
        "schema": EXTRACT_SCHEMA, "repository": repository, "commit": source.commit,
        "prefixes": sorted(prefixes or []), "tree_files": len(tree),
        "weights_loaded": 0, "weight_bytes_read": 0, "upstream_code_executed": False,
        "excluded_by_custody": dict(sorted(excluded.items())),
        "excluded_examples": dict(sorted(excluded_examples.items())),
        "studies": dict(sorted(studies.items())), "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-dir", required=True, type=Path, help="Local clone or bare clone; no checkout needed")
    parser.add_argument("--ref", required=True, help="Ref resolved to an immutable commit SHA")
    parser.add_argument("--repository", required=True, help="owner/name used in URIs")
    parser.add_argument("--prefix", action="append", default=[], help="Limit to repository-relative path prefixes")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--git", default=None, help="git executable")
    parser.add_argument("--raw", action="store_true",
                        help="fetch JSON bytes from raw.githubusercontent.com at the pinned commit (blobless clone)")
    args = parser.parse_args()
    source: ObjectSource = (RawGitHubSource(args.git_dir, args.ref, args.repository, args.git) if args.raw
                            else GitSource(args.git_dir, args.ref, args.git))
    result = extract(source, args.repository, args.prefix or None)
    data = canonical(result) + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as handle:
        handle.write(data)
    print(json.dumps({"commit": result["commit"], "observations": len(result["observations"]),
                      "studies": len(result["studies"]), "sha256": sha256(data)}, sort_keys=True))


if __name__ == "__main__":
    main()
