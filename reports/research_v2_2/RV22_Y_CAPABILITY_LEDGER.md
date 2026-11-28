# RV22-Y: freeze synthetic results and integrate checked public evidence

Parent RV22-X 4062d328 passed both hosted workflows. End the synthetic
memory-controller escalation. All prior reports, implementations and spent
evaluation panels remain historical development evidence; no promotion.

## Capability and failure ledger

| Stage | Durable result | Boundary / failed promotion |
|---|---|---|
| H | Exact routing support audit: 896 vectors, five seeds, zero selective corrections | Synthetic routing labels, not research competence |
| I/J | Decision-value supervision and full policy replay; shortlist candidate | Learned heads trail exact two-step on I; earlier J speed claim depends on slower baseline |
| K | Mask/budget quotient for deterministic positive-cost observations; optimized exact search | No recurrence-specific or adaptive-width speed advantage |
| L | Checked canonical continuation equivalence and replay | Persistent proof DAG 8.18x slower on six-query mixture |
| M/N/O | Root-only caching, economic admission and bounded streaming history | Learned admission loses to simple controls; history/verification overhead |
| P/Q | Actual key bypass and cost-supervised activation | Probe alias and little learned-specific behavior |
| R/S | Invariant fingerprints, Walsh marginal kernel and parity witnesses | Coarse and pairwise collisions; full key mandatory |
| T | DGK/1 bounded binary codec; capacity cliffs; 78.9% key-byte reduction vs JSON | Key bytes, not general RAM or CPU win |
| U/V | Overflow guard, recurrence intervals, retained arena | Thrashing, delayed adaptation and fixed-horizon overestimation |
| W/X | Finite-horizon prior, single-key no-eviction 2-competitive bound, finite offline oracle | Unknown termination and eviction break assumptions; no universal controller |

## Selected integration

New isolated package files _exact_engine.py, exact_research.py and
curriculum_audit.py. Engine extracts the previously tested RV22-P integer
DP, Bellman checker and replay, RV22-S legal test projection and RV22-T
DGK/1 canonical codec; add a 100000-edge fail-closed solve limit. No
fixture generator, learned policy, history sketch or timing heuristic
in the integrated path. Original scripts and frozen v2.0/v2.1 unchanged.

ExactResearch validates/copies JSON-shaped inputs: at most 64 hypotheses,
8 tests, integer budget 0..12, positive integer masses <= 1e6, costs
1..65535, bounded signed rewards, unique restricted ASCII test IDs,
strict 0/1 integers and complete vectors. Reject unsupported inputs.
The default is cold checked solving. Explicit capacity 0..65536 enables
an in-process root-only freeze cache: full binary key equality, independently
checked Bellman value on every miss, original-row replay, no eviction and
no external certificate ingestion. No automatic routing or production
deployment. Capacity counts key plus serialized policy/value bytes only;
not RAM, RSS or a sandbox security boundary. Trusted single-process Python
callers must not mutate internal cache fields. No CPU speed claim.

action_values returns exact forced-first-action continuation, myopic
value, action costs and branch masses. Test cost is subtracted once per
path. Native policy_value validates reachable splits and path costs.
Reviewed native Distill curriculum contract at
MoralityLabAI/Compactification-Theory@7fa19c90372b13d71676e6fef13819d2a18b9be3,
Distill/research_curriculum.py and Distill/research_cases.py. The audit
checks manifest/split digests, public root hashes, public action inventory,
full rational values/regret/one-step labels, forced policies, baseline
ties and aggregate regrets. Reject raw-root duplication and exact
canonical overlap across splits. Bound each JSON file to <1 MiB and
each split to 256 roots. Does not authenticate origin, reconstruct full
packet provenance, validate teacher prose, or establish source checkout
identity; the operator must verify that separately. Upstream custody
remains evaluator_only and training_admission false. No fitting.

## Local evidence

64 fresh independent synthetic roots, each with an invariant variant
(seed 2026760000); 128 exact root value comparisons against a separate
index/used-test oracle, 128 binary key/scale comparisons against RV22-T,
1280 forced-action and one-step comparisons against the independent oracle.
All agreed exactly. 64/128 cache hits, 12134 retained serialized bytes
under explicit 65536-byte allowance. No runtime benchmark. Two local
pytest functions passed, one skipped (native Distill checkout absent).
Native-shaped XOR manifest fixture: 3 roots/15 action records, rejection
of a modified action value even after digests are recomputed, split overlap,
invalid input, repeat policy test and zero capacity fallback. Native
integration test calls upstream build(counts=(2,2,2)) when installed;
native execution and full 96-root export audit remain pending.

Measured local source blobs:
_exact_engine.py 850f7e93ceaa6fa622fdd981bdf8ba8d6a2ea794
exact_research.py ab5216b0f815b3cd114bc4707add1c8b5f6b75b8
curriculum_audit.py b35d6c2cf110a2e6b32240b8303fce0d86d3fc3a
test_exact_research.py 7427a494fed7f018dc023f128a529516df91e1d8

## Next execution gates

On an operator host with reviewed upstream checkout at the pinned commit:
    PYTHONPATH="$DISTILL_CHECKOUT:$DATAGOBLIN/src" python -m Distill.research_curriculum --out "$FRESH_PUBLIC_EXPORT"
    PYTHONPATH="$DATAGOBLIN/src" python -c 'import sys,json; from datagoblin.research_v2_2.curriculum_audit import audit_directory; print(json.dumps(audit_directory(sys.argv[1]),indent=2))' "$FRESH_PUBLIC_EXPORT"

Run the optional native integration test and inspect the 48/16/32 split
audit. A separately authorized supervised-probe export must be defined
before any reference-only rows are used as fit targets. Keep actor
inference restricted to public rows, with evaluator labels held separately.
Compare realized net utility with exact and cheap analytic controls;
use fresh roots and families for any promotion claim.

Outstanding native v2.2 actor comparison: actual model bindings and private
exclusion registry, fresh held-out 64 cases, matched CPU budgets, paired
primary/capability endpoints. Native Jinn/Beast adaptive exit requires
the original private archive and independent 600-root confirmation panel.
Baby Goblin: recover checkpoint/optimizer/scheduler/RNG/cursor/corpus/
tokenizer; independently check learning-rate continuity, prompt/EOS,
forward/loss accounting and held exclusion before 20M. No launch or
50M+ stage authorization. No paid compute, GPU usage, production importer,
sealed data or training admission in Y. Hosted CI pending.
