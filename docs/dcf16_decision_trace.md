# DCF-16 verifier-grounded decision traces

Schema `dcf16.decision_trace.v1`, validated by `schemas/dcf16_decision_trace.schema.json`, loaded by
`datagoblin.datasets.dcf16`, configured in `configs/dcf16_decision_trace.yaml`.

Source (not vendored):
https://github.com/MoralityLabAI/Compactification-Theory/tree/research/dcf01-dense-compute-frontier-20261125/research/dcf16_decision_curriculum
(`data/{train,val,heldout}.jsonl.gz`, `PROTOCOL.json`, `MANIFEST.json`, generator `curriculum16.py`).
A 4-record val and 2-record heldout sample lives in `tests/fixtures/dcf16/`.

## What it is

Each record is one parity query on a synthetic binary factor graph with independent uniform bit priors. It holds the
candidate dependency representations (each flagged `sufficient` or not), the inference operators with exact counted
costs, the selected strategy, the elimination-trace receipts, the final decision or abstention, and the verification
outcome. The reference posterior is computed exactly and confirmed by independent exact paths (`verified_by`). Regret
is measured against that reference. `hard_negative` flags mark records where the unary shortcut, the one-hop
neighbourhood, or belief propagation reaches the wrong decision.

## Supervision views

`supervision_view(records, target)` yields `{id, split, family, n, input, target}` rows. The input carries the problem,
conditions, decision rule, budget, and observation history, never the targets.

| target | label fields |
| --- | --- |
| `dependency_edges` | `edges` |
| `relevant_variables` | `relevant_variables` |
| `complexity` | `log2_exact_cost` |
| `algorithm` | `label` (selected option), `operator` |
| `recurrent_depth` | `recurrent_depth` |
| `calibration` | `p_reference`, `exact_action` |
| `abstention` | `abstain`, `verified_by` |

## Split hygiene

Held-out families `star`, `dense`, `hyper`, `ktree`, `xor_noiseless`, and `weak_dense` never occur in train or val.
`chain`, `grid`, and `sparse` occur in heldout only at sizes 48 and 64; train and val stop at size 32. The loader
raises `SplitHygieneError` when a record's split tag differs from its file, when a held-out-only family or an
oversize record appears in train or val, when an id repeats within a split, or when an id appears in two splits.
`training_views` reads only train and val, and `supervision_view` refuses heldout records unless an evaluation caller
passes `allow_heldout=True`.

## What it is not

- Not natural language: there are no reasoning traces, chain-of-thought text, or prose explanations.
- Not natural data: the factor graphs are synthetic, generated from fixed families, sizes, and seeds.
- Algorithm labels `gf2` and `particle` are absent from train and val, so they test label-space transfer on heldout.
- Posteriors are verified under the solver model; for misspecified records (`reference_model: "generating model"`)
  the reference is the generating model's posterior.
