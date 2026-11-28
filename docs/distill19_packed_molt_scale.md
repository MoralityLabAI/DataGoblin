# DISTILL-19: packed MoLT exact selection scale study

DISTILL-18 preserved native behavior but lost submission CPU on the five-candidate fixture. This additive development-only experiment keeps the reference default and tests a different implementation with immutable public objective preparation once per start observation.

## Packed tie rule

For sorted affordable IDs i=0..n-1, encode each positive utility u_i as the integer (u_i << n) + 2^(n-1-i). For a 0/1 selection the low bits are distinct and cannot carry into the utility field. Maximizing this sum prioritizes utility. Two equal-utility sets cannot be proper subsets because utilities are strictly positive; the smallest differing ID determines lexicographic order. The set containing that ID has the larger low-bit mask, and therefore the larger packed score. Both sparse snapshot and descending dense-capacity DP maximize this score exactly. Prepare immutable validated rows once, filter current exclusions, preserve the native objective, and bind current frame/receipt identifiers on each submission. No cached authority, no source changes to the existing host.

## Frozen paired workloads

Native host: 24 roots per train/calibration and width cell (5,8,12,16), both relevant and unrelated source changes. Width five uses the original native generator; wider arrays use a deterministic development extension of World, with positive costs/utilities and capacity 2*width. The original Episode independently verifies complete trace parity and optimality by its combinations oracle. Four arms: original, DISTILL-18 compact (no cache), packed sparse and packed dense. Total 1,536 native episode executions. Policy CPU includes construction, start preparation and all decisions; host work and observation copying excluded for all arms. Also report submit CPU separately.

Wider public objective workload: widths 16,32,64,128, tight and medium capacity (2*width or 8*width), 16 roots per cell, four distinct source exclusions per root. The exact native reference DP provides paired selection results; verify feasibility and receipt binding as well. This is a public solver replay, not a native host episode. Four arms, 2,048 selection requests. Time includes per-root preparation plus all revisions, excluding copying before the timed block. Rotate arm order by root index. No artificial exact-repeat bank. No timing-based autotuning, protected evaluation roots, model calls or training admission. CPU timing descriptive, small samples and Python/runtime dependent. Record all root outcomes and negative performance results. Keep native host verifier and default planner unchanged.

Tests exhaustively enumerate small feasible subsets with ties, random input order, forbidden candidates and capacities; include 128-bit tie masks, stale/current receipt separation and native-host checks. 
