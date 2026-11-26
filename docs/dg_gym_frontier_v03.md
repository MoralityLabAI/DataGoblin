# DG-Gym v0.3: efficiency frontiers and competence-gated acquisition

Status: public development, no paid compute, no corpus admission, no production launch.

## Comparison cohort

Use the same task distribution, public split, initial checkpoint identity, hardware
profile, budget estimand and objective schema. A receipt stores gain and resource
intervals with a common evidence status (measured or estimated); unknown costs
cannot be inserted as zero. Estimated and measured cohorts are separate. A
measured cohort still needs independently validated telemetry and a registered
uncertainty analysis; this module does not infer either.

For benefit intervals [l,u] and cost intervals [a,b], A robustly dominates B
when every benefit lower bound of A is at least the corresponding benefit
upper bound of B, and every cost upper bound of A is at most the cost lower
bound of B, with at least one strict inequality. Conditional on simultaneous
coverage of these intervals, A dominates B for every compatible realization.
The reverse implication need not hold when the coordinates are correlated.

A feasible frontier includes only competent rows whose cost upper bounds
satisfy the hard caps. Retain nondominated rows, including equal points with
distinct identities. The exact O(n^2) implementation is a small-population
reference. Local tests checked 81 ordered point pairs and all 512 subsets of
a 3x3 benefit/cost grid, plus cap, interval and cohort rejection cases.

An exploratory priority index is

    index = (midpoint(gain) + k * half_width(gain)) / upper_bound(cost).

This is a ranking heuristic, not a calibrated value-of-information estimate.
The next learner should train on past campaigns, hold out entire architecture
and task lineages, compare predictive calibration with empirical family means
and an unlearned priority rule, and use only public-development results for
selection. Sealed outcomes remain outside the prediction inputs.

## Bijective MOD7 follow-up

Each affine instruction r -> a*r+b (mod 7) has a in {1,...,6}. The slope
of any remaining suffix is nonzero, so changing an intermediate residue
changes the exact terminal target. With uniform initial r, conditioning on
the instruction sequence leaves the terminal residue uniform. This rules
out a population shortcut that ignores the initial residue.

The sandbox pilot used the 16,782-parameter GRU, two CPU threads, three
training seeds (11,23,37), 1,200 updates per arm, 96 examples/update. The
length curriculum is 1,2,4 at fractions 1/4,1/4,1/2. Calibration length 5,
development length 6. The competence gate is >=256/512 and at least
52 roots above the larger of the empirical majority and last-additive
heuristics. These are screening heuristics, not the exact symbolic solver.
Every fit made 115,200 draws and 316,800 recurrent example-visits.

| Seed | Task | Concept | Shuffled |
| --- | ---: | ---: | ---: |
| 11 | 78 | 73 | 85 |
| 23 | 61 | 67 | 58 |
| 37 | 58 | 70 | 58 |
| Total /1536 | 197 | 210 | 201 |

Heuristic 83/512. Median training seconds: 2.0659, 2.5207, 2.2102.
The concept-supervised arm gained 13/1536 (+0.85 percentage points);
paired gains -5/512, +6/512, +12/512. Recovery/corruption counts summed
across fits are 55/42, descriptive only. All nine fail competence. The
activation audit did not run. No causal fidelity, peak RAM/VRAM, or
end-to-end hardware efficiency result. These are fixed-exposure fits,
not equal wall-clock fits. No positive learning claim or promotion.

Original local source SHA-256:
50893b8441d5d22017a6c1308063839121b06fe639c8e77eb71e4c7a6fbe2be2
(PyTorch 2.10.0+cpu). The formatted repo continuation has a different
source hash; rerun it to obtain a new source-bound receipt. The local
archive retains the exact executed source and original receipt.

## Next architecture: learned finite-state transition

Use a probability vector p in the seven-state simplex and a learned
row-stochastic matrix P_(a,b) for each instruction. The exact reference
P*_(a,b) is a permutation matrix. Since total variation is contractive
under a stochastic matrix, if every learned row is within epsilon TV of
its exact counterpart, then after K visits the distance from the exact
state is at most min(1,K*epsilon). This follows from telescoping the
per-step transition error and applying contraction. With point initial
state and per-row correct transition probability at least 1-epsilon,
the probability of following the entire correct path is at least
(1-epsilon)^K. The exact affine model has epsilon=0.

Evaluate task loss, state supervision and shuffled controls with identical
transition architecture and exposures. A one-step audit can enumerate
all 7*6*7=294 state-instruction triples. A known state coordinate makes
the interchange intervention exact; a separately learned basis should
be tested against that interpretable reference. This is a proposed next
gym, not a result of the current GRU pilot.
