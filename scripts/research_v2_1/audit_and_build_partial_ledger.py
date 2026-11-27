"""Build the partial Research Verifiers v2.1 ledger from completed Windows receipts and verified WSL logs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from datagoblin.research_v2.suite import FAMILIES, digest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "research_v2_1_controller_persona"
RECEIPTS_DIR = REPORT_DIR / "receipts"

LEAK_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/mnt/c/Users/", re.IGNORECASE),
    re.compile(r"/home/[a-zA-Z0-9_.-]+/"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)


def main() -> int:
    jinn = json.loads((RECEIPTS_DIR / "jinn_beast_disentanglement.json").read_text(encoding="utf-8"))
    bg = json.loads((RECEIPTS_DIR / "baby_goblin_early_specialist.json").read_text(encoding="utf-8"))

    ledger = {
        "schema": "datagoblin-research-v2.1-partial-ledger-1",
        "status": "partial_workstreams_bc_complete_workstream_a_prospective_3_of_5_pending_wsl_recovery",
        "historical_baseline_pr93_commit": "d4aa46c3ff68f03395cc0edadd5faae05782cd84",
        "v20_implementation_sha256": "73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339",
        "v21_implementation_sha256": "149e8082d3115d657880258bb2581f02fe1483d1da62adfee25a562e02b681f3",
        "families": list(FAMILIES),
        "workstream_a_status": {
            "vf_validate_n8": {"gold_valid": 8, "noop_valid": 8, "valid_rate": 1.0},
            "dev_v21_final_suite_sha256": "f6e4010e243562b3b22e39dec566d76ea53be0619f8ecf2e5a48745dfd9d3c62",
            "dev16_controls": {
                "reference": {
                    "report_sha256": "319a8b7709b9e0c277c11c5462171615d01f39bf0e0dceed4df5952dff2c809c",
                    "macro_score": 0.987305,
                    "macro_capability_score": 0.996826,
                    "success_rate": 1.0,
                },
                "noop": {
                    "report_sha256": "27b2c5bb6d5c7d56c1032743e3f6f4c233f1271b4020790056a2f34e64522005",
                    "macro_score": 0.0,
                    "macro_capability_score": 0.0,
                    "success_rate": 0.0,
                },
            },
            "dev8_final_qwen3": {
                "autonomous": {
                    "report_sha256": "c6fb07ecc8e6b508fb0d7cf2dd1154fabfd084f8a934141e7087beba90a8e023",
                    "cases": 8,
                    "macro_score": 0.0,
                    "macro_capability_score": 0.26875,
                    "success_rate": 0.0,
                    "request_failures": 3,
                    "inference_requests": 28,
                    "wall_seconds": 443.588823,
                },
                "rlt_state_correction": {
                    "report_sha256": "37323d7670ef0b565abaea77625fee32666c3ffc589ae923a66fc06dbeebf753",
                    "cases": 8,
                    "macro_score": 0.140137,
                    "macro_capability_score": 0.503784,
                    "success_rate": 0.125,
                    "request_failures": 1,
                    "inference_requests": 19,
                    "total_interventions": 9,
                    "skipped_model_turns": 1,
                    "wall_seconds": 303.908395,
                },
            },
            "prospective_v21_final_suite_sha256": "80a1dbc545adec6dbc95842e8df5889decfe9e6674487956357085fa01c44be0",
            "prospective_v21_final_held_registry_sha256": (
                "f7d6f24037f42097e24178613033848b0c472e2df25aacdcea1eb8cd8ac96a5e"
            ),
            "prospective_32_completed_before_wsl_disk_halt": {
                "qwen3-4b-instruct-2507-q8@autonomous": {
                    "report_sha256": "9b8d0a2691e9f07c525adf203b08403b97bbf45feed4b8bcfd43602d3ea2e792",
                    "cases": 32,
                    "macro_score": 0.03125,
                    "macro_capability_score": 0.1765625,
                    "success_rate": 0.03125,
                    "request_failures": 21,
                    "inference_requests": 80,
                    "wall_seconds": 1392.194233,
                },
                "qwen3-4b-instruct-2507-q8@deterministic_stopping": {
                    "report_sha256": "f4c7f93804c99ae88665c8f037a43ad3cc668982bcc16eb7b5d425d6379da6cc",
                    "cases": 32,
                    "macro_score": 0.03125,
                    "macro_capability_score": 0.128125,
                    "success_rate": 0.03125,
                    "request_failures": 23,
                    "inference_requests": 83,
                    "total_interventions": 1,
                    "skipped_model_turns": 1,
                    "wall_seconds": 1639.478155,
                },
                "qwen3-4b-instruct-2507-q8@learned_ldt_stopping": {
                    "report_sha256": "e47b1d4d07339ab04eb638dea38d3e3d3313a6c60ed0354a8992c35cd3bc4d57",
                    "cases": 32,
                    "macro_score": 0.03125,
                    "macro_capability_score": 0.134375,
                    "success_rate": 0.03125,
                    "request_failures": 22,
                    "inference_requests": 83,
                    "total_interventions": 1,
                    "skipped_model_turns": 1,
                    "wall_seconds": 1573.546075,
                },
            },
            "prospective_32_pending_wsl_recovery": [
                "qwen3-4b-instruct-2507-q8@trm_action_routing",
                "qwen3-4b-instruct-2507-q8@rlt_state_correction",
                "baby-goblin-10m-v2-q4km@autonomous",
                "baby-goblin-10m-v2-q4km@rlt_state_correction",
            ],
            "winner": None,
        },
        "jinn_beast_disentanglement": jinn,
        "baby_goblin_early_specialist_experiment": bg,
    }
    ledger["ledger_sha256"] = digest(ledger)
    out = REPORT_DIR / "benchmark_v21_partial_ledger.json"
    raw = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(raw)

    issues: list[str] = []
    for folder in (
        REPO_ROOT / "src" / "datagoblin" / "research_v2_1",
        REPO_ROOT / "environments" / "datagoblin_research_v2_1",
        REPO_ROOT / "scripts" / "research_v2_1",
        REPO_ROOT / "configs" / "research_v2_1",
        REPORT_DIR,
        REPO_ROOT / "docs" / "research_verifiers_v2_1.md",
        REPO_ROOT / "tests" / "unit" / "test_research_v2_1.py",
    ):
        files = [folder] if folder.is_file() else sorted(folder.rglob("*"))
        for p in files:
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.name in {"transcript.jsonl", "cases.private.jsonl"}:
                issues.append(f"forbidden file: {p}")
                continue
            if p.name in {"export_v21_receipts.py", "audit_and_build_partial_ledger.py"}:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for pat in LEAK_PATTERNS:
                if pat.search(txt):
                    issues.append(f"leak pattern {pat.pattern!r} in {p.relative_to(REPO_ROOT)}")
    if issues:
        raise RuntimeError("Privacy/path audit failed:\n" + "\n".join(issues))
    print(
        json.dumps(
            {
                "status": "ok",
                "ledger_path": str(out.relative_to(REPO_ROOT)),
                "ledger_bytes": len(raw),
                "ledger_sha256": hashlib.sha256(raw).hexdigest(),
                "audit_issues": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
