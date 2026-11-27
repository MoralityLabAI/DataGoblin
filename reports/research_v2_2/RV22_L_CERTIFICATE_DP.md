# RV22-L: canonical full-depth continuation certificates

Status: completed local synthetic CPU study; no controller promotion, no paid
inference or GPU use. Scope: exact finite deterministic tests with positive
integer costs, positive hypothesis masses, binary outcomes, stable test IDs
and orientation, and fixed terminal rewards.

## Construction

For each state remove constant tests and unaffordable tests. With positive
costs and deterministic observations, neither can become informative and
affordable later on that path. Sort legal tests by ID, merge rows with identical
class and remaining test bits, and divide all merged masses by their GCD.
The resulting tuple includes budget, both rewards, test IDs/costs and normalized
rows. Equal keys imply identical legal continuation models, up to a common
positive mass scale. No digest alone authorizes reuse; the full tuples are
compared. IDs and bit orientation are preserved, so this is a sufficient
canonical equivalence, not a general graph-isomorphism algorithm.

Let W be normalized mass and F the integer mass-weighted optimal utility.
For a test of cost c with child canonicalization scales s0 and s1,

    F(K) = max(S(K), max_test(s0 F(K0) + s1 F(K1) - c W(K))).

Strictly decreasing budget makes the proof DAG acyclic. The checker recomputes
the complete Bellman maximum bottom-up for each new record, using only already
checked dependencies, then compiles the selected policy. Every query replays
the policy using original rows and verifies its integer utility after undoing
the normalization. Stable IDs permit row/column permutation and duplicate-row
splitting without changing policy semantics. Changes to affordable costs,
budget or rewards enter the key; class/mass changes may invalidate it unless
the normalized legal observation distribution is still identical.

The prototype trusts its in-process memory; it does not authenticate imported
certificates, prove the implementation in Lean, or guarantee a byte-level RAM
bound. The retention gate caps verified state count at 50,000 and declines
whole newly checked DAGs that would overflow. Temporary solve/check memory and
policy trees are outside that count. No external proof ingestion.

## Campaign

Development: 48 trajectories, 288 queries, exact parity with compressed mask DP.
After measuring replay overhead, compile and memoize selected policies.
Fresh 64 trajectories (32 ordinary 12-hypothesis/6-test, 32 ordinary
16-hypothesis/8-test), six sequential queries per trajectory, new seed 2026112813.
Variants: anchor, row/test permutation plus 3x uniform mass scaling, split one
duplicate row while preserving total mass, increase first test cost by one,
condition on first test's positive outcome and deduct cost, unrelated world.
Memory starts empty per trajectory. Arm order randomized within each query.
Compare cold mask/budget DP with policy compilation/replay, cold canonical
DP with Bellman check/replay, and persistent checked canonical DP. CPU process
time includes construction and replay; all arithmetic value comparisons exact
by integer cross multiplication. All 384 queries matched; no lost utility.

| Query | Mask DP ms | Cold certificate ms | Persistent certificate ms | Root hits |
|---|---:|---:|---:|---:|
| Anchor | .8313 | 12.9159 | 12.8358 | 0/64 |
| Permutation and scaling | .8293 | 13.2988 | .0740 | 64/64 |
| Duplicate split | .8285 | 13.0120 | .0798 | 64/64 |
| Cost mutation | .7290 | 10.7835 | 9.1969 | 0/64 |
| Conditioned | .1762 | 1.8081 | .0695 | 64/64 |
| Unrelated | .7750 | 12.6654 | 11.8539 | 0/64 |

Repeated equivalent queries are 11.2x and 10.4x faster than cold mask DP;
conditioned queries 2.53x. Cost mutation uses 39,596 verified dependency hits
and solves/checks 30,488 edges versus 37,676 cold canonical edges. Dependency
hits count lookup events, not unique states. Unrelated queries have many
small terminal-state hits but almost unchanged edge counts.

End-to-end six-query mean per trajectory: 4.169 ms mask baseline, 64.484 ms
cold certificates, 34.110 ms persistent certificates. The persistent scheme is
about 8.18x slower than compressed mask DP on this mixture. If future queries
are all equivalent permutations, the anchor setup requires about 16 additional
hits to break even versus repeated cold mask DP (ratio of mean costs; not a
distributional guarantee). Neither the workload nor timing is a promotion gate.

Independent regression: 32 fresh roots x 10 queries = 320 exact comparisons
against legacy used-mask full DP, including budget, reward, class and cost
mutations. Exact equality on all; no incorrect root reuse on these mutations.
Capacity=1 declines storage without changing value; reserved action IDs
rejected. Hosted regression added. No independent timing confidence intervals.

## Reproduce

    python scripts/research_v2_2/certificate_dp.py --out /tmp/rv22l.json

Measured source SHA-256 c86ecec312b47542a6b65f994fc3b7901a84707243206263c6cf7f35a6eaaa87
(only two subsequent line wraps; repository SHA-256
85b2d7b20c1dce5dbfc9edaec16e6382fb41529dd36ecb75467364d53b0e5943).
Measured receipt SHA-256 a45be46fa79150951c187d7a20686aa06ddabe65c48b6d6f7a0c7cb9f378cd86.

## Next gate

Decouple cold integer mask DP from certificate publication. Start with cheap
root identity and selected-policy reuse, charge proof extraction/checking as an
investment, compare against plain identity caching and independently verified
proof transport. Track memory bytes, bounded replacement, amortized CPU and
reuse-prediction accuracy. Train a controller on prospective reuse value only
after the cheap analytic retrieval baseline is measured. All measured cohorts
are spent.
