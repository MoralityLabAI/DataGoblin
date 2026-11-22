# MeTTa-07: keyed confirmation with separately held answers

The new evaluator prepares novel synthetic MeTTa fact worlds from a private,
high-entropy key. A deterministic HMAC stream selects the content of each
root. The challenge file contains fact programs and questions, while the
reference answers remain in the scorer's private process. An aggregate-only
score can be produced from model predictions.

**Public CI runs use a fixed dummy key.** They check reproducibility,
challenge tampering, answer redaction and scoring logic. Those public checks
are an engineering test, not a sealed model evaluation or evidence of model
generalization.

## Setup

The private custodian generates a random binary key of at least 32 bytes
outside the repository. Do not commit, print, paste into chat, or attach the
key to a GitHub workflow, issue, PR, or artifact.

```sh
# Create the key using your trusted host's cryptographic random generator.
# Keep it outside Git and restrict permissions to the evaluation custodian.
python MeTTa/training_data/scripts/sealed_finite_eval.py prepare \
  --secret-file /secure/metta07.key \
  --challenge-file /secure/eval/challenges.jsonl \
  --commitment-file /secure/eval/commitment.json \
  --roots 64
```

Share only `challenges.jsonl` with the model evaluation runner. Freeze
`commitment.json` before model inference. Keep the key entirely inside the
custodian. The challenge IDs and root IDs are HMAC digests, and all reference
answers are recomputed from the private key during scoring.

The challenge format uses one JSON object per line:

```json
{"challenge_id":"<digest>","source_prefix":"(Group ...)\n","query":"(match &self ...)","root_id":"<digest>"}
```

The example is illustrative; generated records also include schema and
custody metadata. Each prediction must be submitted as:

```json
{"challenge_id":"<digest>","answers":["(Selected n00)","(Selected n04)"]}
```

Answers are strings containing individual MeTTa expressions. An empty
`answers` array denotes an empty result set. Order is immaterial and
alpha-equivalent variable expressions are normalized. Repeated answers
retain multiplicity. Unknown or duplicate IDs invalidate the submission.

```sh
python MeTTa/training_data/scripts/sealed_finite_eval.py score \
  --secret-file /secure/metta07.key \
  --challenge-file /secure/eval/challenges.jsonl \
  --commitment-file /secure/eval/commitment.json \
  --predictions-file /secure/eval/model_predictions.jsonl \
  --summary-file /secure/eval/aggregate.json
```

The scorer checks byte-identical challenge regeneration and commitment
agreement before reading predictions. It reports only total, correct,
incorrect, missing and malformed counts, plus input digests and the scope
of the reference semantics. Per-challenge judgments and hidden targets
stay inside the private scorer.

## What this measures

The reference evaluator accepts ground facts and
`(match &self pattern projection)` with repeated-variable consistency and
nested projections. It reports exact-root generalization to unseen,
key-generated fact instances *if* the key and answers were withheld before
model inference. The generated query templates are public, so structural
generalization to unrelated MeTTa programs is a separate claim requiring
additional evaluator families.

A result becomes a genuine sealed experiment only after the custodian
attests that the key, source roots and target answers were previously
unseen by the model and its training machinery. The scorer intentionally
sets `external_custody_attested=false` for an unaudited local run.

All source training data remain quarantined. The private evaluator does not
promote the 187 imported author-asserted candidates or turn native agreement
into independent full-MeTTa semantic correctness.

## Rehearsal

```sh
python -m unittest discover -s MeTTa/training_data/tests \
  -p 'test_sealed_finite_eval.py' -v
```

The public CI rehearsal uses a dummy key and synthetic oracle predictions.
It tests positive and mutated negative controls, including an altered
reference answer, missing rows, duplicated IDs, changed commitments and
a wrong key. This validates scoring machinery and never substitutes for an
unpublished model-run receipt.
