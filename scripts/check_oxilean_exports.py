"""Qualify OxiLean on a version-matched native Lean export and substitution controls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file

OXILEAN = "3adb983d2aefc6a596bdc8b63438f54feecd1f55"
EXPORTER = "3de59f10bc4b4a0f2de698597aeb1246caa0df0a"
ROOTS = (
    "dgOxiIdentity", "dgOxiPairSwap", "dgOxiEitherLeft", "dgOxiEitherRight",
    "dgOxiTransport", "dgOxiCompose", "dgOxiUnit", "dgOxiRefl",
)
NEGATIVES = frozenset({"dgOxiUnit", "dgOxiCompose"})


def run_command(args: list[str], directory: Path, environment: dict[str, str], label: str, expected: int = 0) -> str:
    result = subprocess.run(args, cwd=directory, env=environment, capture_output=True,
                            encoding="utf-8", timeout=240, check=False)
    (directory / f"{label}.stdout").write_text(result.stdout, encoding="utf-8", newline="\n")
    (directory / f"{label}.stderr").write_text(result.stderr, encoding="utf-8", newline="\n")
    if result.returncode != expected:
        raise ValueError(f"{label} exited {result.returncode}, expected {expected}")
    return result.stdout


def checked_commit(repo: Path, expected: str) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], encoding="utf-8").strip()
    if head != expected:
        raise ValueError(f"upstream source pin mismatch: {head}")
    return head


def run(lean: Path, exporter: Path, checker: Path, exporter_repo: Path, checker_repo: Path,
        out: Path) -> dict[str, Any]:
    lean, exporter, checker = [path.resolve(strict=True) for path in (lean, exporter, checker)]
    out = out.resolve()
    out.mkdir(parents=True)
    env = {**os.environ, "PATH": f"{lean.parent}{os.pathsep}{os.environ['PATH']}", "LEAN_PATH": str(out)}
    provenance = {
        "oxilean_commit": checked_commit(checker_repo, OXILEAN),
        "lean4export_commit": checked_commit(exporter_repo, EXPORTER),
        "oxilean_cargo_lock_sha256": sha256_file(checker_repo / "Cargo.lock"),
        "lean_binary_sha256": sha256_file(lean), "exporter_binary_sha256": sha256_file(exporter),
        "checker_binary_sha256": sha256_file(checker),
        "lean_version": run_command([str(lean), "--version"], out, env, "lean-version").strip(),
        "checker_version": run_command([str(checker), "--version"], out, env, "checker-version").strip(),
    }
    if "version 4.32.0-rc1," not in provenance["lean_version"]:
        raise ValueError("OxiLean pilot requires the exporter-matched Lean 4.32.0-rc1")
    source = Path(__file__).resolve().parents[1] / "apps/lean-goose/verification/OxiPilot.lean"
    shutil.copyfile(source, out / source.name)
    native = run_command([str(lean), "-o", "OxiPilot.olean", source.name], out, env, "native-valid")
    if any(f"'{name}' does not depend on any axioms" not in native for name in ROOTS):
        raise ValueError("expected every native proof to compile without axioms")
    for name, statement in {
        "dgOxiUnit": "True",
        "dgOxiCompose": "∀ p q r : Prop, (p → q) → (q → r) → p → r",
    }.items():
        control = f"{name}Invalid.lean"
        (out / control).write_text(f"import Init\ntheorem {name} : {statement} := ({statement})\n", encoding="utf-8")
        run_command([str(lean), control], out, env, f"native-control-{name}", expected=1)
    exported = run_command([str(exporter), "OxiPilot", "--", *ROOTS], out, env, "export")
    valid = out / "valid.ndjson"
    valid.write_text(exported, encoding="utf-8", newline="\n")
    objects = [json.loads(line) for line in exported.splitlines() if line]
    format_version = objects[0]["meta"]["format"]["version"]
    if format_version != "3.1.0":
        raise ValueError(f"unexpected export format: {format_version}")
    names = {obj["in"]: obj["str"]["str"] for obj in objects if "str" in obj and obj["str"]["pre"] == 0}
    changed = set()
    for obj in objects:
        theorem = obj.get("thm")
        if theorem is not None and names.get(theorem["name"]) in NEGATIVES:
            theorem["value"] = theorem["type"]
            changed.add(names[theorem["name"]])
    if changed != NEGATIVES:
        raise ValueError("corruption controls did not resolve the expected declaration names")
    corrupted = out / "substituted.ndjson"
    corrupted.write_text("".join(json.dumps(obj) + "\n" for obj in objects), encoding="utf-8", newline="\n")
    results = {}
    totals = {}
    for label, path, exit_code in (("valid", valid, 0), ("substituted", corrupted, 1)):
        checker_report = out / f"checker-{label}.json"
        run_command([str(checker), "--json", str(checker_report), "--json-full", "--limits", "corpus", str(path)],
                    out, env, f"checker-{label}", expected=exit_code)
        data = json.loads(checker_report.read_text(encoding="utf-8"))
        verdicts = {item["name"]: item["verdict"] for item in data["decls"] if item["name"] in ROOTS}
        expected = {name: "rejected" if label == "substituted" and name in NEGATIVES else "verified"
                    for name in ROOTS}
        if verdicts != expected:
            raise ValueError(f"unexpected named declaration verdicts: {verdicts}")
        results[label] = {v: sum(value == v for value in verdicts.values())
                          for v in ("verified", "unsupported", "rejected")}
        results[label]["named_declarations"] = verdicts
        totals[label] = {"verdict_counts": data["totals"], "unsupported_features": data["unsupported_features"]}
    report = {"provenance": provenance, "export_format": format_version, "named_roots": results, "totals": totals}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {path.name: sha256_file(path) for path in sorted(out.iterdir()) if path.is_file()}
    (out / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("lean", "exporter", "checker", "exporter-repo", "checker-repo", "out"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.lean, args.exporter, args.checker, args.exporter_repo, args.checker_repo, args.out)))


if __name__ == "__main__":
    main()
