# DataGoblin research v2 for Verifiers

This package exposes the repository's research session as a Prime Intellect
`verifiers==0.4.1` environment. The versioned research contract is v2; the upstream
Python API is `verifiers.v1`.

Install both packages from the DataGoblin checkout with Python 3.11 or later:

```bash
uv pip install -e . -e environments/datagoblin_research_v2
vf-validate datagoblin-research-v2 --runtime.type subprocess -n 8
```

`vf-validate` runs the public development reference trajectories on CPU and checks
that an untouched session earns zero. It calls no language model. Keep evaluation
manifests outside the checkout and use the core suite loader to check their split.

## Text action protocol

Every model receives the same task text and returns one JSON action per turn.
`ResearchSession.step_json()` runs the bounded experiment and supplies the next
public observation. The model cannot execute arbitrary Python or shell commands.
Only the evaluator's final session report sets the reward.

The environment exports `ResearchTaskset` and `ResearchEnv`. Upstream discovers
them through `__all__`; `vf-run datagoblin-research-v2` selects the custom turn
loop automatically. The package pins the `null` harness and local `subprocess`
runtime. Each episode allows at most eight responses and rejects a response over
16,384 UTF-8 bytes. The CPU experiments have their own resource limits in the core
package. Agent retries default to zero.

The framework records the private task record in its trace. Treat evaluation
traces as private artifacts; publish the aggregate report after checking it.

## Bind an evaluated model

The registry is `configs/research_v2/models.json`. An operator supplies a running
endpoint and records its exact weights, prompt format and inference settings.
The environment does not download weights or start inference hardware.

For an endpoint that already accepts OpenAI-style chat completions:

```bash
vf-run datagoblin-research-v2 \
  -m "$SERVED_MODEL" \
  --client.base-url "$DATAGOBLIN_EVAL_BASE_URL" \
  --client.api-key-var DATAGOBLIN_EVAL_API_KEY \
  --env.taskset.split development \
  --sampling.temperature 0.6 \
  --sampling.max-tokens 1024 \
  -n 8 -r 1 -c 1
```

Set both variables to the operator's chosen endpoint and served model identifier.
Do not omit the endpoint flags: the upstream CLI has its own inference-provider
defaults. Each result must carry the registry entry and endpoint receipt from the
core comparison runner before joining a tournament report. The command above is
an adapter smoke run and does not by itself certify that an endpoint served the
claimed weights.

Verifiers retains the conversation history between turns. Its trace metadata
therefore records `protocol_id="verifiers-history-v2"`. Use the core comparison
runner for the primary `research-v2-context2048` track: that runner refreshes each
prompt and requires an exact token count before generation. Keep these tracks in
separate reports.

Baby Goblin uses raw completion framing. HRM-Text requires a condition envelope
and bidirectional PrefixLM attention over the prompt. The Verifiers `null`
harness speaks chat completions, so those two models need a serving adapter that
renders that framing correctly, or the core completion runner. Record the adapter
and selected HRM condition. A generic chat template can change the measured task.

For evaluation, set `--env.taskset.suite-path` to the operator-held manifest and
`--env.taskset.split heldout`. The package refuses evaluation generation from
its public development seed. Keep the same suite identity, action limits and
sampling policy for every model in the comparison.

## Endpoint binding for the core comparison runner

Save a binding outside the repository and replace every `REPLACE_...` value with
the measured SHA-256 for the exact served artifact. This example is a schema
template; its placeholders deliberately fail validation.

```json
{
  "schema": "datagoblin-research-v2-binding-1",
  "registry_id": "qwen3-4b",
  "model_id": "Qwen/Qwen3-4B",
  "served_model": "Qwen/Qwen3-4B",
  "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
  "weights_sha256": "REPLACE_WITH_EXACT_SERVED_WEIGHT_ARTIFACT_SHA256",
  "tokenizer_sha256": "REPLACE_WITH_EXACT_TOKENIZER_SHA256",
  "template_sha256": "REPLACE_WITH_EXACT_CHAT_TEMPLATE_SHA256",
  "config_sha256": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
  "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
  "precision": "bf16",
  "recurrence": null,
  "protocol": "chat-completions",
  "base_url": "http://127.0.0.1:8000/v1",
  "tokenize_path": "/tokenize",
  "api_key_env": "DATAGOBLIN_EVAL_API_KEY",
  "prompt_role": "user",
  "thinking_mode": "enabled",
  "chat_template_kwargs": {"enable_thinking": true},
  "context_limit": 2048,
  "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0},
  "resource_caps": {"wall_seconds": 600, "ram_mib": 16384, "max_requests": 128},
  "cap_wrapper_receipt_sha256": "REPLACE_WITH_OPERATOR_RESOURCE_WRAPPER_RECEIPT_SHA256"
}
```

`weights_sha256` identifies the exact loaded artifact. For a sharded checkpoint,
hash the operator's canonical manifest of the loaded shard filenames and their
SHA-256 values and retain that manifest with the binding. The native Baby Goblin
entry requires its recorded single-file hash; an expanded or quantized export
needs a distinct registry entry.

The transport accepts HTTPS endpoints and loopback HTTP endpoints. Credentials
come from the named environment variable; setting `api_key_env` to `null` sends
no authorization header. A server with no tokenizer endpoint needs an adapter
before it can enter the primary comparison.

Tokenization is a POST to `tokenize_path` on the same origin as `base_url`. It
receives the exact `model` and `prompt`, or the exact `model`, `messages` and
`chat_template_kwargs` that generation will receive. Its response must contain
`{"count": integer}` for the fully rendered input, including special tokens and
the assistant generation prefix. An optional `model` field must equal
`served_model`. The response count plus the requested generation limit must fit
2,048 tokens. Generation then returns OpenAI-style `usage` with matching
`prompt_tokens`, a bounded `completion_tokens` count and their exact sum in
`total_tokens`. Include reasoning tokens in completion usage.

The primary comparison uses temperature 0, top-p 1 and seed 0 for every model.
Bindings that change these values fail validation. Model-specific sampling
experiments belong in a separately identified track.

Each action reserves two HTTP requests, including tokenization, and makes at
most one inference request. A shared deadline covers both exchanges. The client
refuses redirects, truncated generation, missing token accounting and model alias
mismatches. Failure receipts record which requests were attempted without storing
credentials or the provider's error body.

For raw Baby Goblin use `protocol="raw-completion"`, `prompt_role="raw"` and
`thinking_mode="not_applicable"`. HRM uses `protocol="prefix-lm-completion"`
with those same role settings, plus `hrm_condition="direct"` or `"synth,cot"`
and `prefix_lm_prefill_ones=true`. The transport sends the source-defined condition
envelope to tokenization and generation, then sends an all-ones `token_type_ids`
array for the prefill. Its operator-provided server must implement that attention
contract.

Both checkpoint identity and the server's resource wrapper are recorded as
`operator_attested`. The client cannot establish which weights a remote server
loaded, or prove its host memory and wall limits. Retain the wrapper receipt and
binding next to the operator's server logs.

## Upstream contract

The implementation follows the `Taskset.load`, `TaskData` and
`Agent.interaction` interfaces at upstream commit
`540f40faab8c376b85c6afa2f872683b0ed1d4d1`:

- [Tasksets](https://github.com/PrimeIntellect-ai/verifiers/blob/540f40faab8c376b85c6afa2f872683b0ed1d4d1/docs/tasksets.md)
- [Turn-by-turn agents](https://github.com/PrimeIntellect-ai/verifiers/blob/540f40faab8c376b85c6afa2f872683b0ed1d4d1/docs/agent.md)
- [Custom environments](https://github.com/PrimeIntellect-ai/verifiers/blob/540f40faab8c376b85c6afa2f872683b0ed1d4d1/docs/env.md)
