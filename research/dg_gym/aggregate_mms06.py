"""Aggregate and verify MMS-06 receipts into reproducible Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lbf20m", type=Path, required=True)
    p.add_argument("--lbf20d-4s", type=Path, required=True)
    p.add_argument("--lbf20d-vec", type=Path, required=True)
    args = p.parse_args()

    m20 = json.loads(args.lbf20m.read_text(encoding="utf-8"))
    d4 = json.loads(args.lbf20d_4s.read_text(encoding="utf-8"))
    dv = json.loads(args.lbf20d_vec.read_text(encoding="utf-8"))

    print("## LBF-20M Clock Calibration")
    print(json.dumps(m20["clock_calibration"], indent=2))

    print("\n## LBF-20M Exact Cache Summary (12 roots, 5 trials)")
    for wl, wv in m20["workloads"].items():
        print(f"\n### {wl} (repeat={wv['repeat']}, inner_reps={wv['inner_reps']})")
        print(
            "| Arm | Q Wall (ms) | Delta Wall | Acct (ms) | Q CPU (ms) "
            "| Delta CPU | Msgs | Hits/Total | Evict | Peak B |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm, s in wv["summary_by_arm"].items():
            tot_q = s["total_query_hits"] + s["total_query_misses"]
            print(
                f"| `{arm}` | {s['mean_total_query_wall_s'] * 1000:.2f} | "
                f"{s['wall_delta_pct_vs_disabled']:+.2f}% | "
                f"{s['mean_total_cache_acct_wall_s'] * 1000:.2f} | "
                f"{s['mean_total_query_cpu_s'] * 1000:.2f} | "
                f"{s['cpu_delta_pct_vs_disabled']:+.2f}% | "
                f"{s['total_messages']} | {s['total_query_hits']}/{tot_q} | "
                f"{s['total_evictions']} | {s['max_peak_query_payload_bytes']} |"
            )

    datasets = (
        ("LBF-20D Vectorized Pre-Overfitting (B_CPU=1.5s)", dv),
        ("LBF-20D Overfitting Regime (B_CPU=4.0s)", d4),
    )
    for label, d in datasets:
        print(f"\n## {label}")
        for mode in ("fixed_cpu", "fixed_exposure"):
            print(f"\n### {mode}")
            print(
                "| Arm | Upd | Mean Regret | Delta V vs Ord | All 3 Seeds >0 "
                "| Root SE | Step CPU Ratio | C_tot (s) | Wall (s) |"
            )
            print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
            for arm, s in d["summaries"][mode].items():
                print(
                    f"| `{arm}` | {s['mean_updates']:.1f} | {s['mean_regret']:.4f} | "
                    f"{s['mean_delta_V_vs_ordinary']:+.4f} | {s['positive_all_seeds']} | "
                    f"{s['per_root_delta_V_se']:.4f} | {s['step_cpu_overhead_ratio_vs_ordinary']:.3f}x | "
                    f"{s['mean_total_cpu_s']:.2f} | {s['mean_total_wall_s']:.2f} |"
                )


if __name__ == "__main__":
    main()
