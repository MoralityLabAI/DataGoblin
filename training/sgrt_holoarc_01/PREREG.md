# DG-SGRT-01 preregistration boundary

**Status:** prospective design and public smoke gates only. Confirmatory training
protocol not frozen; no held-out results may be interpreted as a successful
model experiment before a new frozen protocol and independently held seed.

## Claims under test

1. **Geometry fidelity:** a 12D multi-gear base loop closes; a flat 16D
   connection has identity Wilson transport; a planted noncommuting connection
   has nonzero loop holonomy, determinant +1, low orthogonality drift, and
   a gauge-invariant trace/eigenangle spectrum.
2. **Verifier fidelity:** the public demonstrations are matched exactly,
   target information stays outside all proposer calls, and evaluator-only
   full-domain scoring distinguishes generalizing solutions from demonstration
   overfit. Same-set shuffle must preserve candidate-set SHA and canonical
   full-task success.
3. **Candidate coverage (exploratory):** compare curved excursions against
   flat, smooth random, fixed semantic heuristic, repeated edit and rigid
   top-1 under the same root and checker budget. All current policies are
   deterministic proxies; this stage tests plumbing.
4. **Learned effect (future only):** a task-conditioned low-rank sidecar
   improves independently verified whole-task success over the best learned
   non-geometric and extra-refinement baselines at equal wall-time/compute
   and candidate budgets on family-level holdouts.
5. **Path-order effect (future only):** a separately registered adaptive
   task with decisions between visits is needed to attribute performance to
   itinerary history. Fixed-candidate-set canonical selection is deliberately
   invariant under any permutation.

## Public development design

- Four task families with separate grouped root IDs.
- Two-operation 4-bit typed DSL; exact interpreter and 16-input oracle.
- One deliberately corrupted starting program per root; four public examples.
- Default seed 23, 8 roots, 8 candidate checks per non-rigid policy.
- Fixed 12D base and 16D fiber; eight transport microsteps per circuit.
- Root count hard-capped at 128, proposal count at 16. No model downloads,
  server calls, GPUs, asynchronous jobs or paid compute.
- Preserve public inputs separately from evaluator scores. Generated output
  belongs in ignored local runs and is not a training label without a new
  admission decision.

## Controls and estimands

- Geometry: flat identity loss and curved identity loss; orthogonality,
  closure and covector pairing errors. Nontrivial curved holonomy alone is
  a mathematical construction, not task competence.
- Verifier: chosen candidate success over all 16 inputs, any-target-reachable
  candidate coverage, first-valid visit, calls per root and checks per
  verified success. Count zero-solve arms explicitly.
- Matched arms: compare all non-rigid controls at eight candidate checks;
  report rigid top-1 separately with its one actual call.
- Exact-set shuffle: mandatory identical candidate-set hashes, identical
  canonical success under the complete-checker protocol. Differences in
  first-hit positions are a diagnostic, not a solve-rate advantage.
- Costs: actual process time, resident memory and invoked model tokens are
  required for a trained or externally connected pilot. Toy Python loops
  do not stand in for 1B recurrent block compute.

## New confirmation registration required before training

Freeze independent train/development/sealed family and operator-pair splits,
source/repo and scorer hashes, the proposer/adapter parameter count, random
initialization, task prompts, recurrence budget, how task-condition signals
enter the gear and connection, learning schedule, budget and halt semantics,
verifier identity, effect threshold and power analysis. Record all losses,
including nulls. For a model-architecture claim, compare from matched
optimizer snapshots with and without the low-rank sidecar.

Use independent task-root clustered uncertainty estimates, with multiplicity
handling when evaluating several learned variants. Define the minimum gain
from development variance before touching sealed labels. A positive
geometry effect is a prerequisite for mechanistic interpretation, not a
replacement for verified task gain.

## Safety and interoperability

- Keep the independent verifier in control of acceptance and simulator moves.
- A proposer may generate candidates but receives no hidden target, oracle
  action, heldout continuation, or evaluator-derived utility while selecting.
- Do not route this pilot to the running local Inky Mini server. A future
  optional adapter can use the existing server with a 12k default context,
  a 16k hard maximum, pinned endpoint revision/quant/tokenizer/sampler,
  and zero paid fallback.
- Keep Aleph 1B architecture changes behind separate matched checkpoints;
  the current sidecar is CPU mathematics and a program-repair fixture.
