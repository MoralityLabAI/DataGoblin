#!/usr/bin/env python3
"""Pinned, source-only mathematics-corpus accounting for DataGoblin.

This is an intake census, not a Lean kernel verification or proof-success audit.
An Aleph tokenizer is not frozen. Token counts always name the exact encoding used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "Compactification-Theory": ("MoralityLabAI/Compactification-Theory", "071cf29e27a71fff858cc3bbb4dc3ea4193c9b53"),
    "EvalMath": ("MoralityLabAI/EvalMath", "bed334f032d31d777bad0e5152e732f57252092b"),
    "LoopedTransformers": ("MoralityLabAI/LoopedTransformers", "39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a"),
    "BPD": ("MoralityLabAI/BPD", "60dd9ed19aeddd8dddc40a6facad36140a1250ed"),
    "semaev-index-calculus": ("patrickdugan/semaev-index-calculus", "3c630e179ae837dad839949dcb03cf4fdb41815e"),
}
EXTENSIONS = {".lean", ".py", ".md", ".tex"}
DECL = re.compile(
    r"^[ \t]*(?:(?:private|protected|noncomputable|partial)\s+)*"
    r"(theorem|lemma|def|abbrev|instance|example|structure|inductive|axiom|opaque)\b"
    r"(?:[ \t]+([^\s:({]+))?",
    re.MULTILINE,
)
FAILURE = re.compile(
    r"\b(fail(?:ed|ure|s)?|rejected?|counterexample|refut(?:ed|ation)|"
    r"correction|inconclusive|open (?:problem|gap)|blocked|invalid|negative result)\b",
    re.IGNORECASE,
)
ASSUMPTION = re.compile(
    r"\b(assum(?:e|ption)|conditional|hypothes(?:is|es)|unproved|"
    r"missing (?:lemma|proof|step)|conjecture|gap|unverified)\b",
    re.IGNORECASE,
)
MAX_CONTENT_BYTES = 2_000_000


def run(*cmd: str, cwd: Path | None = None) -> str:
    done = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def clone_pinned(root: Path) -> None:
    """Download only selected source extensions; refuse to silently drift SHAs."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (remote, commit) in SOURCES.items():
        dest = root / name
        if dest.exists():
            actual = run("git", "rev-parse", "HEAD", cwd=dest)
            if actual != commit:
                raise ValueError(f"{name} has {actual}; expected {commit}")
            continue
        run("git", "init", "-q", str(dest))
        run("git", "remote", "add", "origin", f"https://github.com/{remote}.git", cwd=dest)
        run("git", "sparse-checkout", "init", "--no-cone", cwd=dest)
        run("git", "sparse-checkout", "set", "--no-cone",
            "*.lean", "*.py", "*.md", "*.tex", cwd=dest)
        run("git", "-c", "protocol.version=2", "fetch", "-q", "--depth=1",
            "--filter=blob:none", "origin", commit, cwd=dest)
        run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=dest)
        actual = run("git", "rev-parse", "HEAD", cwd=dest)
        if actual != commit:
            raise ValueError(f"SHA drift: {name}: {actual} versus {commit}")


def category(repo: str, rel: str) -> str:
    p = rel.lower()
    name = Path(rel).name.lower()
    suffix = Path(rel).suffix.lower()
    if suffix == ".lean":
        if repo == "semaev-index-calculus":
            if name.startswith("instanced"):
                return "generated_lean_instance"
            if name.startswith("chainstandardd"):
                return "generated_lean_template"
            if name.startswith(("mindegreem", "refuted")):
                return "repeated_lean_certificate_wrapper"
            if "axiom" in name or "check" in name:
                return "lean_audit_entrypoint"
            return "general_lean_development"
        if repo == "EvalMath" and p.startswith("fixtures/external/"):
            return "lean_external_import"
        if repo == "EvalMath" and name in {"patchwitness.lean", "patchwitness3.lean"}:
            return "generated_lean_witness"
        if repo == "LoopedTransformers" and "/certificates/" in p:
            return "lean_certificate_fixture"
        if "fixture" in p or "test" in p or "curriculum" in p or "smoke" in name:
            return "lean_fixture_or_curriculum"
        return "lean_research"
    if suffix == ".py":
        if "test" in p or "fixture" in p:
            return "python_tests_and_fixtures"
        if "generate" in name or "export" in name:
            return "generators_and_exporters"
        return "python_research_and_verifiers"
    if suffix in {".md", ".tex"}:
        if p.startswith("dialogue/") or "session_logs/" in p:
            return "research_dialogues"
        if "/literature/" in p or p.startswith("literature/"):
            return "external_references"
        if "/results/" in p or "/checkpoints/" in p:
            return "result_notes"
        if p.startswith("paper/") or p.startswith("papers/") or suffix == ".tex":
            return "mathematical_papers"
        return "research_notes_and_protocols"
    return "other"


def split_role(path: str) -> str:
    parts = set(re.split(r"[/_.-]+", path.lower()))
    if parts & {"test", "sealed", "holdout", "holdouts", "reserved"}:
        return "protected_eval"
    if parts & {"dev", "validation", "val"}:
        return "development"
    if "train" in parts:
        return "training_candidate"
    return "unassigned"


def is_numeric_certificate(kind: str) -> bool:
    return kind in {"generated_lean_instance", "generated_lean_template"}


def strip_lean_comments(text: str) -> str:
    """Remove Lean block/line comments, retaining line numbers for source anchors."""
    result = []
    i = 0
    depth = 0
    line_comment = False
    in_string = False
    while i < len(text):
        two = text[i:i + 2]
        char = text[i]
        if char == "\n":
            line_comment = False
            result.append(char)
            i += 1
            continue
        if line_comment:
            result.append(" ")
            i += 1
            continue
        if depth:
            if two == "/-":
                depth += 1
                result.extend("  ")
                i += 2
            elif two == "-/":
                depth -= 1
                result.extend("  ")
                i += 2
            else:
                result.append(" ")
                i += 1
            continue
        if not in_string and two == "/-":
            depth += 1
            result.extend("  ")
            i += 2
            continue
        if not in_string and two == "--":
            line_comment = True
            result.extend("  ")
            i += 2
            continue
        if char == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        result.append(char)
        i += 1
    return "".join(result)


def decl_rows(repo: str, rel: str, source: str, token_count) -> list[dict]:
    clean = strip_lean_comments(source)
    matches = list(DECL.finditer(clean))
    rows = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
        body = clean[match.start():end]
        # This is a conservative source-level split, not Lean syntax elaboration.
        raw_head = re.split(r":=|\bwhere\b", body, maxsplit=1)[0]
        head = raw_head.strip()
        has_sorry = bool(re.search(r"\b(sorry|admit)\b", body))
        axiom = match.group(1) == "axiom"
        # A candidate cluster fingerprint that deliberately omits the declaration name.
        # Matching syntax still requires review of namespaces, imports and assumptions.
        tail = raw_head[match.end() - match.start():]
        name_free = re.sub(r"\s+", " ", f"{match.group(1)} {tail}").strip()
        rows.append({
            "repository": repo,
            "path": rel,
            "line": clean.count("\n", 0, match.start()) + 1,
            "kind": match.group(1),
            "name": match.group(2),
            "statement_sha256": hashlib.sha256(re.sub(r"\s+", " ", head).encode()).hexdigest(),
            "name_free_statement_sha256": hashlib.sha256(name_free.encode()).hexdigest(),
            "source_lane": category(repo, rel),
            "split_role": split_role(rel),
            "source_tokens": token_count(body),
            "has_sorry_or_admit": has_sorry,
            "is_axiom_declaration": axiom,
            "verification": "source_only_no_kernel_replay",
        })
    return rows


def markdown_candidates(repo: str, rel: str, content: str, token_count) -> list[dict]:
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(r"^#{1,4} ", line)]
    if not starts:
        starts = [0]
    rows = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(lines)
        chunk = "\n".join(lines[start:end])
        nfail = len(FAILURE.findall(chunk))
        nassump = len(ASSUMPTION.findall(chunk))
        if not nfail and not nassump:
            continue
        rows.append({
            "repository": repo,
            "path": rel,
            "line_start": start + 1,
            "line_end": end,
            "heading": lines[start][:160] if lines else "",
            "candidate_tokens": token_count(chunk),
            "failure_marker_count": nfail,
            "assumption_marker_count": nassump,
            "status": "heuristic_candidate_requires_human_review",
        })
    return rows


class TokenCounter:
    def __init__(self, encoding: str, tokenizer_json: Path | None) -> None:
        if tokenizer_json:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
            self._encoding = None
            digest = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
            self.label = f"hf-tokenizer-json-sha256:{digest}"
        else:
            import tiktoken
            self._encoding = tiktoken.get_encoding(encoding)
            self._tokenizer = None
            self.label = f"tiktoken:{encoding}"
    def __call__(self, content: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(content, add_special_tokens=False).ids)
        return len(self._encoding.encode(content, disallowed_special=()))


def jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in entries:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def audit(root: Path, out: Path, tokens: TokenCounter) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files, declarations, candidates = [], [], []
    seen = set()
    group = defaultdict(lambda: defaultdict(int))
    totals = Counter()
    for repo, (_, pinned) in SOURCES.items():
        base = root / repo
        if not base.is_dir():
            raise FileNotFoundError(base)
        if (base / ".git").exists():
            current = run("git", "rev-parse", "HEAD", cwd=base)
            if current != pinned:
                raise ValueError(f"wrong commit in {repo}: {current}")
        for path in sorted(base.rglob("*")):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in EXTENSIONS:
                continue
            rel = path.relative_to(base).as_posix()
            size = path.stat().st_size
            kind = category(repo, rel)
            skip = size > MAX_CONTENT_BYTES or is_numeric_certificate(kind)
            item = {
                "repository": repo, "commit": pinned, "path": rel, "category": kind,
                "bytes": size, "counted_tokens": None, "unique_tokens": None,
                "content_sha256": None, "exact_duplicate": None,
                "exclusion_reason": "generated_certificate_payload" if is_numeric_certificate(kind)
                    else ("file_exceeds_content_cap" if size > MAX_CONTENT_BYTES else None),
                "training_eligible": False,
                "verification": "not_run",
            }
            group[repo]["files"] += 1
            group[repo]["source_bytes"] += size
            group[repo][f"kind:{kind}:bytes"] += size
            group[repo][f"kind:{kind}:files"] += 1
            if skip:
                totals["excluded_files"] += 1
                totals["excluded_bytes"] += size
                files.append(item)
                continue
            data = path.read_bytes()
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                item["exclusion_reason"] = "non_utf8"
                files.append(item)
                continue
            digest = hashlib.sha256(data).hexdigest()
            is_dup = digest in seen
            seen.add(digest)
            count = tokens(content)
            item.update(content_sha256=digest, exact_duplicate=is_dup,
                        counted_tokens=count, unique_tokens=0 if is_dup else count)
            group[repo]["tokenized_tokens"] += count
            group[repo]["unique_exact_file_tokens"] += 0 if is_dup else count
            group[repo]["tokenized_files"] += 1
            group[repo]["duplicate_files"] += int(is_dup)
            group[repo][f"kind:{kind}:tokens"] += count
            totals["tokenized_tokens"] += count
            totals["unique_exact_file_tokens"] += 0 if is_dup else count
            if path.suffix.lower() == ".lean":
                rows = decl_rows(repo, rel, content, tokens)
                declarations.extend(rows)
                group[repo]["lean_declarations"] += len(rows)
                group[repo]["lean_sorry_declarations"] += sum(x["has_sorry_or_admit"] for x in rows)
                group[repo]["lean_axiom_declarations"] += sum(x["is_axiom_declaration"] for x in rows)
            if path.suffix.lower() in {".md", ".tex"}:
                rows = markdown_candidates(repo, rel, content, tokens)
                candidates.extend(rows)
                group[repo]["candidate_sections"] += len(rows)
            files.append(item)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "source_intake_no_kernel_verification",
        "tokenizer": tokens.label, "repos": dict(SOURCES),
        "scope": "Sparse source checkout; Lean Python Markdown TeX; generated Semaev numeric certificates excluded",
        "limits": {"max_content_bytes": MAX_CONTENT_BYTES,
                   "dedup": "full-file SHA256 only, no theorem-ancestry or semantic dedup",
                   "research_episode_candidates": "regex evidence sections only; not accepted trajectories",
                   "train_eligible": False},
        "totals": dict(totals), "per_repo": {name: dict(group[name]) for name in SOURCES},
        "declarations": {
            "source_only": len(declarations),
            "with_sorry_or_admit": sum(row["has_sorry_or_admit"] for row in declarations),
            "axiom_declarations": sum(row["is_axiom_declaration"] for row in declarations),
        },
        "episode_candidate_sections": len(candidates),
    }
    jsonl(out / "files.jsonl", files)
    jsonl(out / "lean_declarations.jsonl", declarations)
    jsonl(out / "reasoning_episode_candidates.jsonl", candidates)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Owned mathematics corpus intake",
        "",
        f"Tokenizer: {tokens.label}. Source-only scan, without Lean compilation or license clearance.",
        "",
        "| Repository | Tracked source bytes | Counted tokens | Exact-file unique tokens | "
        "Lean declarations | Candidate reasoning sections |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SOURCES:
        s = group[name]
        report.append(f"| {name} | {s['source_bytes']:,} | {s['tokenized_tokens']:,} | "
                      f"{s['unique_exact_file_tokens']:,} | {s['lean_declarations']:,} | {s['candidate_sections']:,} |")
    report += [
        "", f"**Overall counted tokens:** {totals['tokenized_tokens']:,}.",
        f"**After exact whole-file deduplication:** {totals['unique_exact_file_tokens']:,}.",
        f"**Source-level Lean declarations:** {len(declarations):,}, "
        f"of which {summary['declarations']['with_sorry_or_admit']:,} contain sorry/admit "
        f"and {summary['declarations']['axiom_declarations']:,} are explicit axiom declarations.",
        f"**Research sections flagged for review:** {len(candidates):,}.",
        f"**Excluded oversized/generated files:** {totals['excluded_files']:,} files / "
        f"{totals['excluded_bytes']:,} bytes.",
        "",
        "The candidate sections are not independent verified trajectories. No proofs were kernel-checked here.",
        "Exact whole-file deduplication does not remove copied partial proofs, derivations, or experiment families.",
        "Generated certificate bytes and test fixtures remain external evidence; admission requires",
        "source licenses, assertion-to-proof checks, root/split lineage, privacy screening, and",
        "a frozen Aleph tokenizer. Token numbers here use the labeled interim tokenizer.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--clone-pinned", action="store_true")
    parser.add_argument("--tokenizer", default="o200k_base", choices=["o200k_base", "cl100k_base"])
    parser.add_argument("--tokenizer-json", type=Path, help="local frozen HF tokenizers JSON; no downloads")
    args = parser.parse_args()
    if args.clone_pinned:
        clone_pinned(args.root)
    if args.out is not None:
        result = audit(args.root, args.out, TokenCounter(args.tokenizer, args.tokenizer_json))
        print(json.dumps({k: result[k] for k in
            ("tokenizer", "totals", "declarations", "episode_candidate_sections")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
