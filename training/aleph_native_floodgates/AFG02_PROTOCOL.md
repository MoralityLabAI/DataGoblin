# AFG-02: trained native Direct-A target, draft competence and cache-depth comparison

Prospective CPU development experiment, following the AFG-01 source fixture
([run 46104294299](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46104294299)).
AFG-01 tested 116 exact FP32 argmax positions and 16 adapter rollouts on
random-weight targets; FP32 and BF16 fixture parity passed. This study
adds actual target learning at reduced width. The 1B native 10M checkpoint
remains a separate operator-supplied qualification, not a substitute target.

## Target, data and competence gate

Use the original frozen Aleph class and original explicit FP32 training
forward, 14 physical blocks, 24 visits, width 32, vocabulary 64, four Q heads,
one KV head, core FF 140. Two independent seeds 137 and 163. The target
learns eight continuation symbols given an eight-symbol prefix. The first
symbol is a period marker (0 or 1); other symbols are sampled independently
from 3..18. The continuation repeats the final one or two prefix symbols.
Shift uses a new marker 2 and period three. All targets are exact finite
generators, not a natural-language competence proxy.

Distinct, deterministic, content-disjoint prefix roots for target fitting
(384), adapter fitting (64), CAL/dev/shift/trace (16 each). The model fits
320 AdamW updates (batch 16, lr .0015, clip norm 1) with uniform final-depth
next-token loss on the eight continuation queries. No CAL-dependent target
stopping or root selection. The declared competence gate is at least 80%
teacher-forced continuation accuracy AND 80% free-running token accuracy
on CAL. Report exact sequence accuracy, free-running token accuracy,
and teacher-forced accuracy on all panels even if the gate fails. Do not
claim a competent-target acceptance result for failed seeds; keep all fits
and outcomes. Shift is a stress panel rather than an expected competence
gate.

## Frozen target and adapters

Use the same native AFG-01 cache implementation. Train two rank-four
draft-only adapters per target, first- and second-pass core KV. The
prelude and answer/coda use their matching target visit caches. Target
weights, both core passes, output head and cache authority remain frozen.
Adapter training uses the AFG-01 target-generated-stream fitter on 64
independent adapter roots for 128 updates, batch one, paired mask/LoRA
initialization and minibatch indices. Record all teacher, prefix recompute,
and fitting work. For each cache mode compare the learned adapter with its
original untrained zero-update adapter (same initial mask and rank).

CAL selects a length from 1, 2, 4 for each frozen adapter, minimizing the
declared decode tariff 4*serialized block calls + token-block visits;
tie by smaller length. CAL also selects a length for a cheap deterministic
suffix repeater, which uses committed repetition only, and invokes no
draft model. Ordinary serial target decoding is the baseline. The
repeater is a synthetic-task-relevant control, not a general language
drafter; account for its host decisions separately.

Every rollout must equal independently cold-checked greedy target decoding
exactly. Native draft proposals are followed by original 24-visit verifier
calls and physical KV rollback. Separate target truth accuracy, draft
acceptance, target-call reduction, serialized work, block-token work,
logical cache bytes and decoder-only wall/CPU diagnostics. No assertion
that a structural work reduction translates into GPU speedup.

## Inventory and audit

Default two targets, four trained adapters, twelve policy cells (serial,
suffix repeater, two trained and two initial adapters per seed). 192
dedicated trace rollouts over 16 prefix roots. Model seeds and policies
are paired views, not additional independent roots. Report paired deltas
against serial, repeater and corresponding untrained adapter. All final
panels remain outside training and calibration. Export frozen weights,
root and source hashes, training receipts, CAL choice and exact trace
tapes. Audit refits target and trained adapters, regenerates references,
replays CAL selection and all traces, checks complete inventory and
aggregates. Timing is diagnostic and excluded from exact replay hashes.

CPU only, one PyTorch thread, FP32. No actual Baby Goblin weights,
pretraining claims, GPU allocations or training admission. All records
remain training_ready=false. The purpose is to qualify the native
multi-layer computation on a learned algorithm before attempting real
checkpoint drafting. A failed gate is an honest scientific outcome.
