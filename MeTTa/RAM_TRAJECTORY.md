# MeTTa-11 CPU trajectory pilot

MeTTa-11 uses the MeTTa-10 source allocation and the same Python synthetic
curation oracle. Each TRAIN source presentation now includes the complete
canonical repair from its public initial selection through a checked STOP.
The SFT and RLT files contain the same presentation identities. Each root
contributes one presentation and consumers load one alternate view. Training
on this bundle remains a synthetic research fit under DataGoblin's corpus policy.

The direct and recurrent arms learn a shared scorer over candidate edits
using public per-record and state features. The recurrent arm unrolls a tied
tanh transition and trains it with NumPy backpropagation through time. Both
arms share the reference depth-four matrix multiply-accumulate training
budget, with the direct arm spending it on more optimizer updates. Every fit
uses the same TRAIN roots, initialization seed pair, and maximum of eight
repair decisions. Fixed inference probes at 2, 4, 6, and 8 reuse the same
checkpoint. A separately trained binary head proposes STOP using a threshold
calibrated on TRAIN. The optional allocator learns from cases in which the
eighth pass corrects the second pass's action. Its additional training,
readouts, and recurrent passes are accounted for separately.

The LDT-style proxy projects a proposed edit onto a typed open membership
slot with domain `{excluded, included}`. A proposal narrows that slot to one
value. The synthetic environment checks the type, structural applicability,
and public constraints before applying the edit. Additions require a valid
result, and removals cannot introduce a new violation type. Each decision
opens a fresh membership slot against the current public selection. The
upstream HybridTRMLDT interface informed the typed proposal, claimed
provenance, and independently certified result. The native execution in
this pilot queries ground MeTTa transition facts. The NumPy proxy is the
executed lattice-style controller.

The feedback condition fits on clean and checked veto contexts, then reruns
the network after a veto with the rejected edit and public violation types.
The control uses the same clean/counterfactual training rows and masks,
and falls back to the existing ranking at inference. Each case receipt
counts proposal checks, underlying constraint checks, exact optimum
enumerations, accepted post-veto edits, and retry cost. Each condition
records the number of post-veto edits that reduce repair distance.

## Reproduce

```sh
python -m pip install -e ".[dev,rlt-pilot]" "hyperon==0.2.10"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python -m MeTTa.ram_hillclimb_trajectory --out runs/metta11-dev --native-replay
```

Each run writes into a new directory. The runner caps Linux virtual address
space at 8 GiB by default. Windows records peak working set through
`psutil`. The CLI limits data and optimizer steps independently of the
memory ceiling. The report distinguishes the reference action budget from
actual action work, STOP optimization, and extra allocator steps. Its
checkpoint hashes identify trained weights and frozen thresholds.

A `--proposals` JSON file contains a list of up to eight objects with
fields `arm`, `field`, and `value`. The registry accepts one change at a
time. Each candidate refits from the same seeds and source allocation.
Development selection first checks for an increase in unsafe or invalid
edits. Among eligible candidates it favors verified completion with STOP,
fewer premature stops, first-edit accuracy, remaining distance, and cost.
The development report includes every comparison and its decision.

After development, rerun the identical command with a new output directory
and `--frozen-development-report runs/metta11-dev/report.json`. The runner
checks reproduced source allocations, champion configurations, and
checkpoint digests before generating any conflict tasks. Transfer cases and
aggregates are saved in separate `frozen_conflict_*` files. The five public
development families and the conflict transfer share this synthetic curation
task. Seeds within a family produce independent source roots. Repeated
initializations are paired observations of the same development roots.

## Future adapter contracts

An adapter owns three interfaces. `observe(root)` returns a stable
source-root identity and source hash, a serializable public initial state,
and a typed legal action schema. `verify(state, proposal)` returns an
acceptance decision, resulting public state, and a reason with provenance
and the cost of independent verification. `grade(root, state, stop)` is
an environment-owned terminal check which can read protected labels and
reports verified completion, premature STOP, unsafe transitions, and
optional remaining distance. A fit receives only TRAIN receipts. Every
adapter declares the split before loading its evaluation roots and supplies
its own instance, optimizer, verifier call, time, and memory budgets.

| Adapter | Public proposal | Independent checker and terminal receipt |
| --- | --- | --- |
| Lean | Apply a tactic to a proof state or propose STOP | Sandbox Lean and record kernel-checked proof completion, tactic failures, and timeout |
| Sudoku | Assign a digit to an open cell or propose STOP | Check row, column, and box constraints and verify a completed grid against the clues |
| Boolean-world | Propose a typed state assignment or action | Execute in an independent truth-table or simulator backend and compare the terminal predicate |
| MCP routing | Propose a typed tool call or a STOP decision for a fixture | Verify the tool allowlist, schema, and tool trace against a fixture-owned outcome |
| Native LDT | Propose a typed lattice narrowing against a versioned public state | Use HybridTRMLDT's environment-owned membrane, recording monotonicity, bottom, soundness, and role-specific costs |

Each row is an interface for future experiments. The pilot's results cover
the executed synthetic curation gym and report native MeTTa fact replay
separately from model training and transfer.

## Recorded CPU pilot

The initial run at `5008361` used 15 training roots, 32 reference action
updates, and two initializations on the same ten development roots. Each
arm completed zero repairs. The expanded run used 40 training roots,
200 labeled states, and 64 reference action updates. It retained the same
development roots, seeds 1101 and 1102, and eight decision limit.
Hyperon 0.2.10 matched all 200 MeTTa transition facts. The expanded run
took 17.31 seconds wall time and peaked at 251.70 MiB with native replay.

| Expanded development | Verified and stopped | Premature STOP | Exact first edit | Unsafe adds | Proposal checks | Action passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct control | 3/20 | 12 | 3 | 10 | 80 | 65 |
| TRM control | 0/20 | 8 | 4 | 7 | 104 | 384 |
| Selected TRM | 3/20 | 12 | 5 | 0 | 88 | 284 |
| LDT proxy control | 0/20 | 9 | 4 | 0 | 107 | 376 |
| Selected LDT proxy | 2/20 | 11 | 6 | 0 | 117 | 400 |

These are 20 paired observations over ten roots. The controls share the
action training budget. The selected TRM adds 32 allocator optimizer
updates per fit. Its development improvement comes with more premature
STOPs overall. MeTTa-10 reached 0/10 on the same development roots with
a smaller optimizer budget and a different definition of verifier queries.

The eight proposal steps, in order, were:

| Arm | Field | Value | Decision |
| --- | --- | --- | --- |
| direct | `avoid_revisits` | `true` | Rejected. Same completions and four more premature STOPs |
| trm | `avoid_revisits` | `true` | Accepted. Two completions |
| ldt_proxy | `avoid_revisits` | `true` | Accepted. Two completions |
| trm | `inference_depth` | `2` | Rejected. Both completions lost |
| ldt_proxy | `feedback` | `true` | Rejected. One completion lost. Both arms had one progressing post-veto edit. Feedback required four extra score evaluations |
| trm | `adaptive_depth` | `true` | Accepted. Three completions, 48 fewer passes and ten fewer proposal checks than fixed-depth TRM with revisit filtering |
| direct | `supervision` | `"initial"` | Rejected. First-edit accuracy rose from 3 to 10 while completions fell from three to zero and unsafe adds increased |
| ldt_proxy | `stop_fpr` | `0.05` | Accepted. Two fewer premature STOPs, same completions, and 14 more proposal checks |

The trained champion digests reproduced before frozen conflict evaluation.
Each arm completed 0/4, over two conflict roots and two seeds. The bounded
search ended after eight proposals. These results support development
selection on curation roots. Transfer remained unverified.
