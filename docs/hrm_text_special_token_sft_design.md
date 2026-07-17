# Later special-token SFT experiment design

This document is a design for a later milestone; no special tokens are added in the current implementation.

The proposed study would add separately initialized tokens for SCOUT, ALCHEMIST, and RAT only after establishing a
text-only role-conditioning baseline. Training examples would retain executable validator labels and public structured
rationales, never hidden chain-of-thought or sealed evaluation material. A factorial comparison would hold data,
optimizer, update count, context, and inference budget fixed across textual-role, special-token, and neutral controls.

Primary outcomes would be held-out decision accuracy, violation-mask F1, calibration, selective risk, and transfer to
unseen mutation families. Mechanistic measurements would remain exploratory. Token embeddings and recurrent-state
differences would not be interpreted as roles without causal interventions and recurrent non-HRM baselines.

Go/no-go gates are: tokenizer compatibility, deterministic data manifests, disjoint held-out families, bounded training,
predeclared stopping rules, no sealed tuning, and complete resource/cleanup receipts. A null or adverse result remains a
valid outcome.
