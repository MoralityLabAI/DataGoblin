# DG-ARCH-02: source reread versus damping

## Status and bounded independent probe

2026-11-25. Base 52fa6ed1b194ac6012c7d590c883e83517ef0069.
The network-isolated CPU runtime could not fetch the frozen mixed-data ZIP.
We ran a distinct public synthetic development probe with 21 complete fits:
seven arms, seeds 137/163/191, width 16, 52,944 trainable parameters, 14 physical
GQA/RoPE/SwiGLU blocks, 96 AdamW updates, 8 roots (24 views) per update,
learning rate .001, gradient clipping 1. Same initialization and batches
within seed, same block-visit budgets within depth. The 3-pass groups use
34 rather than 24 applications. Torch 2.10.0 CPU; one thread per fit;
169.912 summed fit seconds, 532,224 nonpadding positions, 15,054,336
block-position applications. No GPU or accelerator credits.

Each root has four bits, two Boolean operators (XOR/AND/OR), a correct old
answer, and three views: unchanged, relevant edit of first premise, irrelevant
edit of fourth bit. Two operator compositions (XOR then OR, OR then XOR)
are held out of training. Exact exhaustive evaluation covers 32 held roots
(96 views), including 16 roots with changed answers; two values of an irrelevant
bit are within the same underlying logical structure. Familiar evaluation
exhausts the 112 roots from the remaining compositions, which overlap the
training support. Exact oracle is 100%; copy-old baseline achieves 83.33%
decision accuracy and 50% joint on held roots. No confidence interval for
population generalization is implied by these finite, dependent roots or
the three training seeds. No text LM loss is measured.

| Arm | Held decision NLL | Held joint | Held changed-answer joint | Familiar joint |
|---|---:|---:|---:|---:|
| direct | .41293 | .55208 | .37500 | .67262 |
| prelude_reread | .60107 | .48958 | .25000 | .64881 |
| site_damp | .69871 | .48958 | .18750 | .61905 |
| global_full_damp | .33017 | .53125 | .56250 | .75595 |
| direct3 | .56688 | .51042 | .25000 | .61905 |
| pass_rlt3 | .46821 | .50000 | .16667 | .64286 |
| pass_damp3 | .48861 | .51042 | .16667 | .57440 |

Native seed-level (held NLL, joint success), seeds 137/163/191 respectively:

| Arm | 137 | 163 | 191 |
|---|---|---|---|
| direct | .29544 / .65625 | .47475 / .46875 | .46861 / .53125 |
| prelude_reread | .67955 / .53125 | .67418 / .46875 | .44946 / .46875 |
| site_damp | .81154 / .50000 | .83378 / .46875 | .45080 / .50000 |
| global_full_damp | .39219 / .50000 | .36227 / .50000 | .23604 / .59375 |
| direct3 | .61470 / .59375 | .54302 / .43750 | .54293 / .50000 |
| pass_rlt3 | .50591 / .59375 | .42963 / .50000 | .46910 / .40625 |
| pass_damp3 | .32669 / .56250 | .52979 / .46875 | .60935 / .50000 |

Native global damping checkpoints scored .33017 NLL and .53125 joint;
runtime swapping them to direct gave 1.25925 NLL and .32292 joint.
Direct checkpoint swapping to reread gave .46275 NLL and .50000 joint.
Three-pass RLT checkpoints swapped to direct3 gave .42342 NLL and .52083 joint.
These are training/runtime interactions on a weak finite proxy. Native
global damping leads NLL and changed-answer joint; direct leads all-root joint.
Prelude reread does not earn promotion, despite better mean NLL than
same-site damping. Three-pass RLT provides no resolved advantage over
constant damping. All arms are near the copy-old baseline, so competence is
insufficient for production selection. No scale-up or paid launch authorized.

Replay each cell with:
    python training/aleph_rlt_placement/source_reread_probe.py --seed 137 --arm direct --steps 96 --width 16 --out runs/source-probe/direct-137.json

Use all seven names in the table and seeds 137,163,191. The standalone
probe script is separate from the production and mixed-study classes. It
does not save weights, and its arithmetic and finite evaluation are limited
to the above synthetic language. The original execution script SHA-256:
e941ce39f76e933fa815ceb5e2a6124646773f4d6cba21b2b8ce044ca7e006d1.
Formatting of the standalone source predates the repository lint pass.

## Mixed-corpus continuation (prospective, not executed here)

The patched original gym supports `prelude_reread` and `site_damp`. Both
read a same-position causal state once, at zero-based executed visit 16
before physical core block 6 on pass 2. The prelude arm reads p; the site
control reads the start of pass 2. Coefficient beta=.25, all other updates
native. This is exactly the proposed InkyPair forward for r=2; r=1 has
no site and r>2 keeps the same single intervention rather than repeating it.
Both receive equal parameter and block-visit counts. The trace records
the extra displacement separately. The original self-test is extended;
the integrated mixed-corpus code and its self-test still require execution
on the connected workstation.

Fetch the evidence commit as in RESUME.md. Use a NEW output directory
because gym.py changed and the old study receipts bind source hashes.
Do not overwrite the earlier RLT fits. On the connected machine run:

    D=training/aleph_rlt_placement
    python "$D/gym.py" --smoke
    python "$D/cpu_worker.py" --wall 10800 study.py --pack runs/placement/mixed --out runs/source-reread/study --width 80 --steps 192 --r 2 --arms direct global_full_damp prelude_reread site_damp --seeds 137 163 191 --roots 48 --probe-roots 6
    python "$D/cpu_worker.py" --wall 10800 study.py --pack runs/placement/mixed --out runs/source-reread/study --width 256 --steps 192 --r 3 --arms direct pass_rlt pass_damp --seeds 137 163 191 --roots 48 --probe-roots 6

Execute the width-256 matrix only after confirming host time and memory.
Choose the two-pass finalist on competent development fits. Refit direct and
the best matched control at the same larger width and token budget. Preserve
seed/root paired rows, training/runtime crossovers, source-use interventions,
throughput, and all nulls. For reread, source swap control: same site and
coefficient, substitute pass-start state. For a source-specific claim require
gain over both direct and site damping on independent premise-edited roots
with matched work. For global damping require transfer to checked research
tasks and text, not only a loss improvement on the synthetic proxy.
Use frozen author-independent families for the confirmation, untouched by
the prior public corpus. Preregister effect size and cluster-aware analysis
before opening those labels. Compare depths with measured equal-resource
budgets as well as within-depth contrasts.

## 10B decision HOLD

InkyMini.md proposes A=direct and B=prelude_reread at r=2; fullscale.json
still proposes B=pass_rlt at r=3. Neither is currently confirmed. Treat both
as candidate records; retain fullscale.json as historical proposal rather
than silently overwriting it. Current priority: two-pass direct and
global_full_damp; keep source reread and same-site damping as experimental
arms, and finish the registered three-pass width-256 replication. Freeze
exactly one challenger only after the mixed and independent confirmation
gates, tokenized production manifest, and measured TPU resource checks.
Keep the 30B campaign and external organelles unchanged. No launch on
the strength of this finite probe.
