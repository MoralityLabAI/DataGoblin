# RLT training and routing in DataGoblin

Use RLT as a selectable feedback schedule for a continuous answer state.
Pair it with a task-specific proposer, a bounded retriever, and an independent
verifier. Keep direct feedback and coefficient-matched damping in the
promotion comparison. Assign the schedule per skill and verifier contract,
using held-out coverage and total work to choose it. Use direct as the
initial learned baseline. Prefer the cheapest policy that preserves
verified coverage across the tested scopes.

## Route a skill request

| Stage | Routing decision | Evidence to retain |
|---|---|---|
| Verify reuse | Reuse a prior certificate when the current goal, environment and dependencies pass an equivalence check. Recheck against the current evidence epoch before committing. | Old and current goal fingerprints, equivalence check, dependency identity, verifier calls. |
| Native tool | Run the skill's inexpensive native solver or automation under its own bounded budget. Stop on a verified answer. | Inputs, tried actions, partial outcomes, tool and kernel work. |
| Grounded fallback | On a native stall, retrieve candidates from the current environment. Run the validated ranker within its evaluated goal families and use lexical order as a baseline and an available fallback. | Candidate pool, retrieval provenance, selected order, visits, candidate checks and abandoned proposals. |
| Refine soft answers | Consider RLT or matched damping when a module revises a continuous answer and testing shows better verified performance or lower total cost. Keep path traversal and other necessary transport updates direct unless a matched experiment supports a change. | State placement, planned clock, realized visits, seed, weights, verified accuracy and work. |
| Commit | Let the independent checker bind the exact answer to the current goal. Stop at the first safe commitment. Leave unresolved candidates as proposals and route remaining work according to the request's remaining budget. | Typed commit or abstention, verifier identity, current epoch, certificate and resource receipt. |

The rule-change pilot explains the placement choice. Interpolating the
one-edge reasoning state suppressed long routes. Interpolating the answer
preserved transport. With verified exits, direct and RLT each solved all
192 edited long conditions and direct used fewer visits. On long no-ops
RLT reconfirmed the answer in one second-turn visit, while certificate
reuse let both policies skip that turn's neural work. The graph verifier
already computes the exact route. Its traversals belong in the cost
comparison. [Rule-change and certificate experiments](rlt_research_track.md#live-ldt-stopping-on-rule-changes)

An RLT module starts a refinement segment with planned depth \(K\) and
uses \(\eta_j=1/(K-j)\) at zero-based visit \(j\). A verifier can stop
the segment after any readout. The router gives the planned clock to
the module and attributes work at the realized depth. This keeps the
comparison between learned schedule, early stopping, and certificate
reuse interpretable.

In the Lean ranker, candidate state starts at zero and the readout is
linear. At the first visit, a schedule multiplies every candidate score
by the same positive coefficient for fixed weights. The initial ranking
is therefore invariant to the runtime schedule. Schedule-dependent training
can still produce different weights. A late-visit or training benefit
needs a comparison beyond that first readout.

The frozen Sudoku TRM shows why ranking the terminal model alone gives a
different decision. Answer-state damping solved 20 of 24 at the terminal
segment and RLT solved 19. Direct finished with 14 and found verified
solutions for 22 earlier in its rollouts. The eight-visit byte prediction
replication supports training with interpolated feedback and testing
compatible inference schedules. Matched damping tied or beat RLT on the
RLT-trained weights across both corpora and three seeds. [Sudoku and language-model experiments](rlt_research_track.md)

## Train and update modules

1. Register the task, candidate grammar, feedback placement, verifier and work
   budget before collection. Make an action eligible as a training positive
   when it independently certifies against the exact current goal. Keep
   genuine misses as negatives and resource or verification uncertainty in
   separate fields. Record every observed attempt, including lexical and
   native controls.
2. Keep the proposal pool and public input representation shared across the
   comparison. Fit small rankers from matched initializations and minibatch
   plans, with supervision at each readout that can be used for an early
   stop. Keep direct, RLT and matched damping on equal planned depth and
   terminal replacement. Record the realized visit count and all verifier
   calls at inference.
3. Separate fresh instances, withheld structures and hardware constraints in
   the release evidence. Compare native automation and lexical ranking at
   equal proposal budgets. Measure the full-pool ceiling separately so a
   missing candidate is visible independently of ranking performance.
4. Preserve an immutable record of the promoted skill and a prior version.
   Replay previous certified and failed goals with fresh withheld groups
   before updating the router. Test both the fitted schedule and runtime
   swaps when deciding whether a result belongs to training or inference.
   Store evaluations separately from examples that may enter a future
   training batch.

## Decision for Lean

Use the native portfolio first and a compact, checked direct ranker for
grounded fallback in the tested scope. Keep lexical order as a cheap
fallback when a fit is unavailable and as the matched control for
promoting future rankers. In the checked training pilot, each fitted
schedule recovered 28 of 60 native stalls on its first fallback attempt,
across both seeds. Lexical recovered four within six attempts. All three
learned schedules used 220 fallback tactic requests and 28 explicit
kernel checks. The shared pool yielded no proof for the other 32 goals,
so prioritize wider candidate generation for that failure mode.
[Checked ranker experiment](rlt_research_track.md#train-the-grounded-ranker-on-checked-outcomes)

Keep RLT available for answer refinement across revisions and for
matched training experiments. The Lean scores support investing in
verified labels and candidate coverage before spending extra recurrent
visits on this ranker. Replay wider task families before routing the
ranker to new proof domains.

For ongoing skill building, report a vector of verified completion, native
stalls recovered, retrieval coverage, verifier work and memory. A worker
can collect new labeled candidates without updating the active ranker.
Promotion is a reviewed decision tied to a capability and its evaluation
scope. Broader claims require evidence across skill families, independent
fits and held-out workloads.
