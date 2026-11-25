# ZKL-01/02/03: synthetic certificate-cost pilot (2026-11-25)

Status: exploratory single-seed local CPU experiment. Upstream reference: [zkLean](https://github.com/rain-1/zero-knowledge-lean/tree/d5279fcfb0f3583bc98dd04f0ae57256c6821f46).
No license is declared upstream. All code and generated tasks here are original; do not ingest upstream source or examples without permission. No generated datasets, weights, or private evaluation seeds are committed. This is a public pilot, not a corpus-release decision or a sealed evaluation.

## Reproduce

With Python 3.10+ and optional CPU PyTorch 2.10:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python training/zklean_certificate_pilot/pilot.py
    python training/zklean_certificate_pilot/normalize.py

The synthetic recurrent ranker uses 889 parameters, 640 training roots, 18 epochs, and a 128-candidate search budget per theorem. It trains on exact unit-cost unfolded Horn tree costs, using premise embeddings and a shared GRU, then evaluates at 3, 6 and 12 loops. Two disjoint seed ranges give 256 ordinary 12-atom and 256 size-shift 24-atom problems. The deterministic verifier checks every returned tree. The exact DP solution is a ground truth for this unit-cost tree grammar; it is not a lower bound on Lean/ZFC proof length or the minimum shared DAG.

| method | same-size solved / optimal / attempts | size-shift solved / optimal / attempts |
|---|---:|---:|
| arity | 254 / 173 / 1819 | 252 / 150 / 2655 |
| DP-2 | 255 / 244 / 934 | 256 / 230 / 1102 |
| DP-5 | 256 / 256 / 755 | 256 / 255 / 824 |
| recurrent-3 | 256 / 226 / 855 | 256 / 224 / 944 |
| recurrent-6 | 256 / 255 / 757 | 256 / 255 / 819 |
| recurrent-12 | 256 / 253 / 758 | 256 / 252 / 823 |
| exact DP | 256 / 256 / 755 | 256 / 256 / 818 |

All denominators 256. Exact DP remains superior; candidate attempts do not measure total time or ranking cost. The current pilot gives no evidence for recurrent scaling advantage. Next preregister independent generator families with deeper distractor chains, cost-weighted rules, matched loop/DP compute, multiple seeds and external evaluator custody. Preserve a separate sealed test set before model selection.

Normalization: a separately typed simply typed lambda microbenchmark yields identical beta-normal forms using eager and normal-order reduction. For 128 nested discarded identities the costs are 129 and 1 beta steps; for required identities both cost 128. This is neither a Lean weak-head evaluator nor a measured dependent certificate saving.

Local run completed; full repository tests, lint and independent steward review remain outstanding. No paid compute or external model checkpoint was used. Keep this isolated as research and use the repository's operator resource rules for future runs.
