# MeTTa-09: local Bonsai / recurrent-model submission adapter

This adapter connects the MeTTa-07 private-key challenge generator to a
locally served OpenAI-compatible chat completion endpoint. Its job is to
produce prediction JSONL suitable for the **separate custodian's** finite
reference scorer, with a compact model/provenance receipt. It never reads an
evaluation key or a reference answer.

## Trust boundaries

The evaluator's private keeper generates a random key and freezes
`commitment.json`, retaining the key on a trusted host. The runner receives
only the `challenges.jsonl` file. That file contains synthetic source facts
and queries whose answers are withheld. The runner sends these prompts to a
**loopback-only** inference endpoint and writes its proposed MeTTa answers.

The key, ground-truth answers and individual correctness verdicts must stay
out of model requests, third-party logs, Git, CI and chat messages. Private
challenge inputs should also stay off external inference services. Use an
SSH tunnel terminating on `127.0.0.1` when model weights live elsewhere,
with explicit access controls for the forwarded port.

All CI jobs for this adapter use a fake in-process transport and a published
dummy key. They check parsing, negative controls and response formatting.
They never load Bonsai model weights or contact a model server.

## Prepare and submit

Use the custodian workflow in
[SEALED_FINITE_EVAL_PROTOCOL.md](SEALED_FINITE_EVAL_PROTOCOL.md) to generate
the challenge bundle. Start a compatible local model server with the desired
checkpoint, then invoke the adapter on the inference host:

```sh
python MeTTa/training_data/scripts/local_model_submit.py \
  --challenges /secure/eval/challenges.jsonl \
  --predictions /secure/eval/bonsai8b_predictions.jsonl \
  --receipt /secure/eval/bonsai8b_receipt.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model-id bonsai-8b-local \
  --model-revision pinned-local-checkpoint \
  --context-window-tokens 12000 \
  --max-completion-tokens 384 --max-cases 256
```

For a 27B or other recurrent checkpoint, supply its model ID/revision and
the appropriate local serving endpoint. To use a 16K context ceiling, set
`--context-window-tokens 16384`. The context-window argument is an advisory
limit unless the serving API returns exact `prompt_tokens` in its usage
metadata. Cases whose reported prompt tokens plus reserved completion budget
exceed the ceiling are marked invalid.

Optionally provide `--model-manifest-sha256` with a measured checkpoint
manifest hash. The adapter records this as user-supplied metadata and always
retains `model_weights_independently_verified=false`. Match that hash to
trusted storage before claiming a pinned model comparison.

Model responses must be one JSON object:

```json
{"answers": ["(Selected n01)", "(Selected n04)"]}
```

The empty set is `{"answers":[]}`. Invalid JSON, malformed response
objects, transport failures and over-budget outputs produce `answers:null`;
this keeps a failed inference from receiving accidental credit on an
empty-answer reference case.

Prediction rows contain only the opaque challenge ID and the model's
proposed expression list. The separate case receipt records request/response
hashes, server-reported model identity, token usage and failure status.
Prompt contents and reference labels are excluded from receipts.

## Private scoring

Return the prediction JSONL to the custodian and run:

```sh
python MeTTa/training_data/scripts/sealed_finite_eval.py score \
  --secret-file /secure/metta07.key \
  --challenge-file /secure/eval/challenges.jsonl \
  --commitment-file /secure/eval/commitment.json \
  --predictions-file /secure/eval/bonsai8b_predictions.jsonl \
  --summary-file /secure/eval/bonsai8b_aggregate.json
```

Commitments and exact challenge bundle hashes are checked before scoring.
A real sealed result additionally needs an external custody attestation
that the evaluation key and reference answers were unseen by the model and
its training pipeline.

The client records model inference **submissions**, not inference gains.
The synthetic mock CI can demonstrate correct scoring of fake predictions
without establishing any real Bonsai performance. Keep actual private
prediction and scoring files out of GitHub Actions artifacts.
