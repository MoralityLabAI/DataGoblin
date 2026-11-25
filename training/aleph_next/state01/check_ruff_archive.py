"""Enforce repository Ruff lint while preserving exact historical replay sources."""

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE = Path(__file__).with_name("RUFF_ARCHIVE_BASELINE.json")


def blob_sha(path):
    raw = path.read_bytes()
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def main():
    baseline = json.loads(BASELINE.read_text())
    expected_version = "ruff " + baseline["ruff_version"]
    version = subprocess.check_output(["ruff", "--version"], text=True).strip()
    if version != expected_version:
        raise SystemExit(f"Expected {expected_version}, got {version}")
    result = subprocess.run(["ruff", "check", ".", "--output-format=json"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise SystemExit(result.stderr or f"Ruff exited {result.returncode}")
    findings = defaultdict(list)
    for diagnostic in json.loads(result.stdout):
        path = Path(diagnostic["filename"]).resolve().relative_to(ROOT).as_posix()
        findings[path].append(diagnostic)
    archived = 0
    failures = 0
    for path, diagnostics in sorted(findings.items()):
        counts = dict(Counter(d["code"] for d in diagnostics))
        pin = baseline["files"].get(path)
        if pin and blob_sha(ROOT / path) == pin["blob"] and counts == pin["diagnostics"]:
            archived += len(diagnostics)
            print(f"Archived: {path}: {len(diagnostics)} exact-source findings ({counts})")
            continue
        failures += len(diagnostics)
        for d in diagnostics:
            row = d["location"]["row"]
            col = d["location"]["column"]
            print(f"{path}:{row}:{col}: {d['code']} {d['message']}")
    print(f"Ruff: {archived} pinned historical findings; {failures} unapproved findings.")
    print("Historical findings are retained, not fixed. Every other diagnostic fails CI.")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
