# ZK-DG-09: graph frontier, scheduling oracle and native CI

Date: 2026-11-27. The fixed-order allocator and frontier R1CS compiler are
implemented; the new hosted tests and native smoke are separate gates.
No new cryptographic proof or native Lean result is claimed.

## Register lifetime

Let sigma be the authorized topological order. Output i is live after step t
if sigma(i) <= t < last(i), where last(i) is the position of its final
consumer. Retain the unique sink until after the final step. Every parent is
read before register writes, so a slot whose last consumer is the current
job may hold the current output immediately afterward.

Let K = max_t |F_t|, including the retained sink. With unit-size outputs
and this read-before-write schedule, K is necessary to hold distinct live
outputs under the explicit storage/no-recomputation model. It is sufficient:
retire every expired interval, then assign the lowest available slot to
the newly produced output. This is an interval coloring of the known
lifetimes. It is not a lower bound on arbitrary compressible or recomputed
values.

The state is (pc, two plan-context limbs, K*w register coordinates).
Layout, graph, weights, parent reads and output depth come from the
authorized plan; all selectors are constrained one-hot against pc. For slot
r define write[r] and keep[r] as sums of disjoint selectors, with keep
requiring the same live owner before and after the selected step. Constrain

    R_next[r] = keep[r] * R[r] + write[r] * h_final

coordinatewise. The selected slot is overwritten; preserved slots retain
their values; retired/unoccupied slots become zero. Induction from the
all-zero initial register file establishes the same sink as the dense
compiler after exactly N steps. The exact terminal register file, context
and step count must be checked by the verifier. This is an argument for
the intended relation, not a formally verified compiler.

## Constraint accounting

For N workers, width w, padded depth D and peak K slots:

    B = 5 + N + w*(N + 61) + D*w*(w + 63)
    dense rows = B + 20*N*w
    frontier rows = B + 21*K*w
    dense arity = 3 + N*w
    frontier arity = 3 + K*w

These counts follow from the existing clamp gadget (61 constraints with
19-bit slacks), register range checks (18 rows per coordinate), and
two vs three rows for a dense vs frontier register update.
They exclude Nova's recursive augmentation and do not measure native
proving time. At w=2, D=4:

| N | Topology | K | Dense state | Frontier state | Dense rows | Frontier rows |
|---:|---|---:|---:|---:|---:|---:|
| 8 | chain | 1 | 19 | 5 | 991 | 713 |
| 8 | previous two | 2 | 19 | 7 | 991 | 755 |
| 8 | all-to-sink | 7 | 19 | 17 | 991 | 965 |
| 32 | chain | 1 | 67 | 5 | 2023 | 785 |
| 32 | previous two | 2 | 67 | 7 | 2023 | 827 |
| 32 | all-to-sink | 31 | 67 | 65 | 2023 | 2045 |

The star graph is a negative control. Compact storage is not always the
cheapest relation; its extra conditional update costs can outweigh the
register reduction. Also, the frontier statement exposes only the sink
and zeroed retired slots, while the dense version exposes all worker
outputs. These are different public-output contracts. Use the dense
version or a separately constrained commitment to intermediate outputs
where downstream corpus admission needs those outputs.

## Ordering optimization

For a predecessor-closed set S, let L(S) count completed vertices with
at least one uncompleted successor, plus the completed terminal if any.
Use subset dynamic programming for N <= 16:

    dp(empty) = 0
    dp(S union {v}) = min over ready v of max(dp(S), L(S union {v}))

Equivalently relax each reachable S by each ready v. The frontier count
depends only on S, so the minimum peak is achieved by optimal substructure.
For larger graphs greedily choose the ready vertex with minimum next
frontier, ties by original index; retain the original plan if the proposed
ordering fails to improve K. The result is only a proposal: any order
change requires the relying party to authorize the new plan and context.

Independent enumeration of the seven-worker two-join example gives 80
topological orders, with minimum K=3 versus K=4 in the breadth-first
input order. This enumeration was executed separately from the Python
compiler. Use the exact DP as a training-label oracle for future TRM/BPD
controllers. It is not a learned routing result. The optimization ignores
parallelism, placement, latency, proof system selection and recomputation.

## Regression gates and native smoke

The preceding commit fdd6e67e passed 760 repository tests with 13 skips
on hosted run 45961200477; Ruff failed with eight reported diagnostics.
Those eight diagnostics have been addressed without suppressing lint.
The new tests/test_zk_frontier.py has 37 cases (24 graph/width fixtures,
4 constraint-count comparisons, 5 single-wire corruptions, terminal
and context checks, bundle checks, and two order tests).
They check every step in both Pasta fields against the dense oracle,
zeroed retired slots, stable shape, state chaining, corruption, and
the exact scheduling optimum against exhaustive enumeration.
New tests are committed; a passing run is required before promotion.

    PYTHONPATH=src python -m pytest -q tests/test_zk_frontier.py
    PYTHONPATH=src python experiments/zk_dg_nova/smoke.py --out /tmp/frontier --nodes 8 --depth 4 --frontier

The scoped GitHub Actions workflow zk-dg-nova.yml attempts a native
release build and a three-node frontier prove/verify/adversarial smoke
with a 25-minute job limit. It preserves the generated Cargo.lock and
public statement/proof artifacts if available. This is a first compile
and execution gate, not a benchmark or proof of security. Local Rust,
Lean and GitHub clone access are unavailable; native success must be
observed in hosted CI or another capable machine. Keep the PR in draft.

## Further work

A committed sparse register file and authenticated intermediate-output
accumulator could preserve corpus lineage while allowing slot retirement.
Separately, a heterogeneous fixed-shape graph would benefit from folding
over a single generic job relation, instead of regenerating parameters
for each authorized graph. Both require explicit binding of the schedule,
model and public-output policy. The current compiler is public integer
arithmetic; no full transformer, model privacy, Lean ZK or live Lucid
deployment is claimed.
