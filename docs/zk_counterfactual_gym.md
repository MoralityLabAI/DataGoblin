# ZKL-18: paired counterfactual exploration teacher

ZKL-17's reference-support teacher produced 231 exploit and 7 explore labels;
all learned policies collapsed to exploitation. ZKL-18 instead compares
verified completion on cloned search states. Every second behavior step,
when the two candidates differ, fork both actions and apply two common
continuation schedules (periodic exploration and seeded 50% exploration)
with identical per-pair RNG seeds. Horizon at most 24 offered columns
including the proposed action. Score zero if no certificate, otherwise
H+1-additional columns. Sum the two score differences, mask ties.
This is a bounded policy-dependent advantage label, not an optimal
action proof. All solved branch outcomes are independently verified
using the pinned SatsHedgeZero checker at 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb.
Behavior trajectories follow a separate 50% random gate regardless of
labels. Clone mutable index, basis bookkeeping, pending sets and lazy
cache; share read-only supports and stored basis vectors.

Training: 8 roots 18001–18008 on five named families over F7/F11, two
behavior trajectories per task, max 48 offers per trajectory. The hard
product-chain-zero family deliberately occurs in both training and test
with separate seeds; no blinded topology generalization claim. The
other three test families do not occur in training. Report collection
CPU, fork pairs, offered columns, coefficient work and actual verifier
calls separately. Fit the existing 177-parameter RNN and 12-parameter
linear gate with inverse-frequency binary label weights (mean weight
one), plus unweighted RNN as an ablation. Identical architecture and
initialization; both training passes use the same counterfactual labels.
The ZKL-17 default training remains unchanged.

Evaluation: 64 tasks at roots 18101–18108 and 18201–18208 on F11/BN254,
eight arms (indexed, row random, periodic, random gate, balanced linear,
balanced RNN, reset RNN, unweighted RNN), three protocols (24 and 96
columns, 10 ms cooperative process-CPU deadline with 96-column cap).
1536 evaluation runs; two campaigns fit deterministic identical models
independently. Actual elapsed CPU includes setup, scoring and verification.
Late verified certificates receive zero credit. 512 MiB virtual-address
cap and whole-process RSS receipt. No model calls, GPU, canonical TRM/LDT/BPD
training, per-case Lean replay or production authorization.

## Hosted result

[Workflow 46032902517](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46032902517)
passed both blocks and all eight branch workflows. Per deterministic fit:
80 trajectories, 1074 steps, 962 fork pairs, 24672 fork columns, 937 solved
branch verifications, 212 exploit and 28 explore labels. At 96 columns,
indexed/periodic/random-gate/unweighted recurrent certified 40/43/42/46
out of 48. The unweighted RNN also credited 46 under the 10 ms deadline;
its CPU cost exceeded the simple controls. Balanced recurrent and reset
certified 43 and 44 at 96 columns. Reset used balanced weights, so there
is no matched memory ablation for the winning policy. The following
ZKL-19 campaign supplies one and compares signed advantage regression.
