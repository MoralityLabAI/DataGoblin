"""Score exported Baby Goblin arm generations inside Linux ResearchSessionV21 (with RLIMIT workers)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datagoblin.research_v2.suite import digest
from datagoblin.research_v2_1.controllers import ResearchController
from datagoblin.research_v2_1.environment import ResearchSessionV21
from datagoblin.research_v2_1.suite import make_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    gens = json.loads(args.generations.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in make_suite(seed=int(gens["eval_seed"]), per_family=2, split="development")}
    rlt_ctrl = ResearchController("rlt_state_correction")

    for arm_name, rows in gens["arms"].items():
        auto_p, auto_c, auto_fits, auto_chk = [], [], 0, 0
        ctrl_p, ctrl_c, ctrl_fits, ctrl_chk = [], [], 0, 0
        for row in rows:
            case = cases[row["task_id"]]
            txt = row["gen_text"]

            sa = ResearchSessionV21(case)
            sa.step_json(txt)
            ra = sa.result()
            auto_p.append(ra["primary_score"])
            auto_c.append(ra["capability_score"])
            auto_fits += int(ra["milestones"]["real_optimizer_fit"])
            auto_chk += int(ra["milestones"]["verified_check_passed"])

            sc = ResearchSessionV21(case)
            post = rlt_ctrl.post_model_decision(
                case=case,
                state=sc.compact_state,
                turn_index=0,
                raw_text=txt,
                artifacts=sc._artifacts,
                remaining=sc._remaining(),
            )
            sc.step_json(post.executed_text or txt)
            pre2 = rlt_ctrl.pre_turn_decision(case=case, state=sc.compact_state, turn_index=1)
            if pre2.skip_model_call and pre2.executed_text:
                sc.step_json(pre2.executed_text)
            rc = sc.result()
            ctrl_p.append(rc["primary_score"])
            ctrl_c.append(rc["capability_score"])
            ctrl_fits += int(rc["milestones"]["real_optimizer_fit"])
            ctrl_chk += int(rc["milestones"]["verified_check_passed"])

        n = max(1, len(rows))
        report["arms"][arm_name]["evaluation"]["verified_research_progress"] = {
            "eval_cases": len(rows),
            "autonomous_mean_primary_score": round(sum(auto_p) / n, 4),
            "autonomous_mean_capability_score": round(sum(auto_c) / n, 4),
            "autonomous_real_optimizer_fits": auto_fits,
            "autonomous_verifier_checks_passed": auto_chk,
            "rlt_controller_mean_primary_score": round(sum(ctrl_p) / n, 4),
            "rlt_controller_mean_capability_score": round(sum(ctrl_c) / n, 4),
            "rlt_controller_real_optimizer_fits": ctrl_fits,
            "rlt_controller_verifier_checks_passed": ctrl_chk,
        }

    report.pop("report_sha256", None)
    report["report_sha256"] = digest(report)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        k: v["evaluation"]["verified_research_progress"] for k, v in report["arms"].items()
    }
    print(json.dumps({"report_sha256": report["report_sha256"], "verified_research_progress": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
