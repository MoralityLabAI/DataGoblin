# Lean-CGRT harness replay

The Claude Code Lean-CGRT c4 run compared flat-S, loop-S and TRM-4M under one Lean
verifier budget. The three-seed means were 27.9%, 27.0% and 24.7% at close@8. The
source tactic closed 564 of 588 certifiable Mathlib states. TRM plus automation
was 1.2 percentage points below automation alone. The archived study and its
[result commit](https://github.com/MoralityLabAI/LoopedTransformers/commit/8739f9ab6d049f532cd5ca18294c0574f4c4d546)
retain the 400M no-go.

The harness gives the first four check slots to the archived automation order.
ALEPH-1B and Bonsai 27B each get an arm with up to four candidate IDs for the
remaining slots. In the cascade, ALEPH gets two slots and Bonsai gets two.
Bonsai sees the attempted ALEPH tactics that lacked a valid closing receipt.
Each arm stops at its first verified closure. The `automation8` control uses
the next four tactics in the original order.

Model packets contain printed Lean goals, candidate IDs and tactic text. They
omit source-reference tactics unless those are already independently present in
the fixed automation portfolio. The kernel receipt stays inside the evaluator.
The archived Lean 4.19.0 certificate must record closure, kernel acceptance and
only allowed axioms. The evaluator accepts IDs from its menu, and never executes
model-generated text. It records local model aliases, calls and reported token
usage. Endpoint aliases come from `/v1/models` and do not attest loaded weights.

## Verified archive replay

The 600 joined source positions reproduced c4's original encoder digest. Of
these, 588 were certifiable. Four checks closed 300 goals and eight closed 307.
The capped menu drawn from archived proposals contains a certified tactic for
27 of the 288 goals left after four checks. Seven of those 27 were also closed
by the next four automation tactics. Across all 588 goals the menu permits a
ceiling of 327 closures, or 55.6%, if a ranker always finds a witness within its
budget. This is historical replay with outcomes already known to the evaluator.
The output has `training_eligible: false` and the command does not export any
training examples. Keep the cohort out of ALEPH's training and selection data.

| c4 split | Certifiable goals | Automation close@8 | Automation misses with a menu witness |
| --- | ---: | ---: | ---: |
| File | 194 | 123 | 1 |
| Long | 196 | 101 | 5 |
| Family | 198 | 83 | 14 |
| Total | 588 | 307 | 20 |

## Run locally

Locate the original c4 `states.jsonl.gz` and the corresponding `mathlib-c3`
directory with `test_file`, `test_long` and `test_family` split traces. Start
with a replay of the archive and the joined goal digest.

```bash
python -m datagoblin rlt-gym lean-harness \
  --states /path/to/evalM_tlm4_trm_s17_29_43/states.jsonl.gz \
  --states-sha256 a38f2207b0bb6e8878b1217b2a2efc01ebe9888dc02b4c6a75fa13795ba22797 \
  --contexts /path/to/mathlib-c3 \
  --encoder-sha256 2c999c28c9f74ab0909d7add28af7145283530ede21cff30e5e2c1652b9dc4ad \
  --out runs/lean-harness/c4-audit
```

Use a fresh `--out` and add `--bonsai-port 18130 --bonsai-model bonsai-27b-ptq1`
when the Claude Code Bonsai endpoint is available. It used the PrismML
`Ternary-Bonsai-2-27B-PTQ1_0.gguf` checkpoint. Add `--aleph-port PORT
--aleph-model MODEL` after the proposed 1B Data Goblin checkpoint is served.
Request both ports to enable the cascade and both solo arms. Keep model runs
on local loopback. No model weights or archived goal texts belong in Git.

This replay measures ranking on existing candidate tactics. A fresh proof trial
would instead use the original Lean-CGRT source-position verifier on new
candidates and a newly frozen, independently sampled cohort. The RAM-18 source
gate and the RLT-03 quarantined labels stay outside this replay.
