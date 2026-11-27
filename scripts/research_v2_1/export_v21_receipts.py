"""Export scrubbed, byte-verified Research Verifiers v2.1 receipts and ledgers for Git publication.

Copies immutable report.json files byte-for-byte alongside scrubbed telemetry
receipts (omitting local filesystem paths, raw transcripts, and private held
cases) and verifies zero leak markers across the exported tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datagoblin.research_v2_1.runner import read_report

LEAK_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/mnt/c/Users/", re.IGNORECASE),
    re.compile(r"/home/[a-zA-Z0-9_.-]+/"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)


def _scrub_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_obj(v) for v in obj]
    if isinstance(obj, str):
        s = re.sub(r"[A-Za-z]:\\Users\\[^\"'\s]+", "<scrubbed-local-path>", obj)
        s = re.sub(r"/mnt/c/Users/[^\"'\s]+", "<scrubbed-local-path>", s)
        s = re.sub(r"/home/[^\"'\s]+", "<scrubbed-local-path>", s)
        return s
    return obj


def _export_run(run_dir: Path, dst_dir: Path) -> dict[str, Any]:
    report, _rows = read_report(run_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Preserve immutable report.json byte-for-byte (report.json has zero local paths)
    raw_report = (run_dir / "report.json").read_bytes()
    (dst_dir / "report.json").write_bytes(raw_report)
    entry: dict[str, Any] = {
        "candidate_id": report["candidate_id"],
        "controller_mode": report.get("controller_mode"),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": hashlib.sha256(raw_report).hexdigest(),
    }
    meta_dir = run_dir.parent / f".meta-{run_dir.name}"
    if meta_dir.is_dir() and (meta_dir / "telemetry.json").is_file():
        tel = json.loads((meta_dir / "telemetry.json").read_text(encoding="utf-8"))
        tel_scrubbed = _scrub_obj(tel)
        tel_bytes = (json.dumps(tel_scrubbed, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (dst_dir / "telemetry.json").write_bytes(tel_bytes)
        entry["telemetry_sha256"] = hashlib.sha256(tel_bytes).hexdigest()
    return entry


def _audit_tree(root: Path) -> list[str]:
    issues: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"transcript.jsonl", "cases.private.jsonl"}:
            issues.append(f"forbidden file exported: {p}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in LEAK_PATTERNS:
            if pat.search(text):
                issues.append(f"leak pattern {pat.pattern!r} matched in {p}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-run", type=Path, action="append", default=[])
    parser.add_argument("--dev-run", type=Path, action="append", default=[])
    parser.add_argument("--control-run", type=Path, action="append", default=[])
    parser.add_argument("--jinn-disentangle", type=Path, default=None)
    parser.add_argument("--baby-goblin-exp", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "datagoblin-research-v2.1-receipts-manifest-1",
        "prospective": [],
        "dev": [],
        "controls": [],
    }
    for run in args.prospective_run:
        manifest["prospective"].append(_export_run(run, out / "prospective" / run.name))
    for run in args.dev_run:
        manifest["dev"].append(_export_run(run, out / "dev" / run.name))
    for run in args.control_run:
        manifest["controls"].append(_export_run(run, out / "controls" / run.name))
    if args.jinn_disentangle and args.jinn_disentangle.is_file():
        data = _scrub_obj(json.loads(args.jinn_disentangle.read_text(encoding="utf-8")))
        raw = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (out / "jinn_beast_disentanglement.json").write_bytes(raw)
        manifest["jinn_beast_disentanglement_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.baby_goblin_exp and args.baby_goblin_exp.is_file():
        data = _scrub_obj(json.loads(args.baby_goblin_exp.read_text(encoding="utf-8")))
        raw = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (out / "baby_goblin_early_specialist.json").write_bytes(raw)
        manifest["baby_goblin_early_specialist_sha256"] = hashlib.sha256(raw).hexdigest()

    (out / "RECEIPTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    issues = _audit_tree(out)
    if issues:
        raise RuntimeError("Privacy/path audit failed:\n" + "\n".join(issues))
    print(json.dumps({"status": "ok", "out_dir": str(out), "audit_issues": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
