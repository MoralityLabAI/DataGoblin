"""Export scrubbed, privacy-safe Research Verifiers v2 receipts and bindings into the repo.

Explicitly copies ONLY non-sensitive summary receipts (`report.json`, sanitized
`telemetry.json`, and sanitized binding manifests). NEVER copies `cases.private.jsonl`,
`transcript.jsonl`, or private Jinn/Beast raw rows/policies. Also asserts zero local
filesystem paths or credential strings exist anywhere in exported files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FORBIDDEN_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/mnt/[a-z]/Users/", re.IGNORECASE),
    re.compile(r"/home/[a-z0-9_-]+/", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
)


def _scrub_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "base_url" and isinstance(v, str) and v.startswith("http://127.0.0.1:"):
                out[k] = "http://127.0.0.1:18000/v1"
            else:
                out[k] = _scrub_obj(v)
        return out
    if isinstance(obj, list):
        return [_scrub_obj(x) for x in obj]
    if isinstance(obj, str):
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(obj):
                raise ValueError(f"forbidden path/token pattern matched in exported string: {obj[:120]}")
    return obj


def _assert_clean_text(text: str, label: str) -> None:
    for pat in FORBIDDEN_PATTERNS:
        m = pat.search(text)
        if m:
            raise ValueError(f"forbidden pattern {pat.pattern!r} found in {label}: {m.group(0)!r}")


def export_run(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing report.json in {run_dir}")
    # Preserve exact byte-for-byte report.json so report_sha256 stays independently verifiable
    report_text = report_path.read_text(encoding="utf-8")
    _assert_clean_text(report_text, str(report_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(report_text, encoding="utf-8")

    meta_dir = run_dir.parent / f".meta-{run_dir.name}"
    if meta_dir.is_dir():
        tel_path = meta_dir / "telemetry.json"
        if tel_path.is_file():
            tel = _scrub_obj(json.loads(tel_path.read_text(encoding="utf-8")))
            tel_text = json.dumps(tel, indent=2, sort_keys=True) + "\n"
            _assert_clean_text(tel_text, str(tel_path))
            (out_dir / "telemetry.json").write_text(tel_text, encoding="utf-8")
    return json.loads(report_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="name=path pairs")
    parser.add_argument("--binding-dir", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    exported = []
    for item in args.run:
        name, raw_path = item.split("=", 1)
        rep = export_run(Path(raw_path), args.out_root / "receipts" / name)
        exported.append({"name": name, "candidate_id": rep["candidate_id"], "report_sha256": rep["report_sha256"]})

    if args.binding_dir is not None and args.binding_dir.is_dir():
        b_out = args.out_root / "bindings"
        b_out.mkdir(parents=True, exist_ok=True)
        for p in sorted(args.binding_dir.glob("*.json")):
            obj = _scrub_obj(json.loads(p.read_text(encoding="utf-8")))
            txt = json.dumps(obj, indent=2, sort_keys=True) + "\n"
            _assert_clean_text(txt, p.name)
            (b_out / p.name).write_text(txt, encoding="utf-8")

    print(json.dumps({"exported_receipts": exported}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
