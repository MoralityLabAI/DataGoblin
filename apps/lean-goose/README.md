# Goose Lean rollouts for DataGoblin

`lean-goose` is a Goose GDK application that sends bounded tactic requests to an existing local llama.cpp server. It keeps one GDK worker and uses the provider's loaded model. The DataGoblin runner owns the theorem registry, Lean proof states, verification, and data export.

The default alias and port match the local Inky Mini rollout gym. Its observed checkpoint is `TML-d24-varloop-research-IQ4_NL.gguf`, which llama.cpp reports as 4.5 bits per weight. The launcher accepts an explicit loopback port and model alias. Model identity comes from llama.cpp metadata. Record the checkpoint digest and model launch command separately when they are available.

## Local run

Build the app and install DataGoblin from the repository root:

```bash
cargo build --release --locked --manifest-path apps/lean-goose/Cargo.toml
python -m pip install -e .

python -m datagoblin.training_gym.rlt_gym.lean_goose \
  --lean /path/to/lean-4.12.0/bin/lean \
  --worker apps/lean-goose/target/release/lean-goose \
  --port 8097 --model inky-mini-varloop --out runs/inky-lean-open
```

Use `lean.exe` and `lean-goose.exe` on Windows. The runner uses serial requests, greedy decoding, at most six visits per theorem by default, and a 15,999-token limit that includes the reserved completion. Goose's OpenAI request formatter feeds llama.cpp template and tokenizer preflight. The runner requires equality with provider prompt usage before submitting a proposed tactic to Lean.

The model receives the public statement, accepted prefix, current Lean goals and prior rejection. A visit submits one action. When a reply contains a list of allowed tactics, the runner tries its first action and requests the next action after checking the resulting state. The original reply stays in `transitions.jsonl`. Lean compiles a successful proof again with an explicit axiom audit. The certificate records the proof source digest and verifier digest.

## Export

`sft.jsonl` contains chat `messages` for verified actions on training roots of this pilot. Every row points to its complete proof certificate and the model receipt. Take this file as a small candidate input to a chat-SFT recipe, with lineage-level deduplication before combining it with other corpora. `transitions.jsonl` includes rejections and development visits. `episodes.jsonl`, `provenance.json`, sources and `SHA256.json` allow inspection and replay. The three training and three development roots are purpose-built and modest in difficulty. Their split applies only to this fixture. Their ancestry relative to other training corpora or to the provider's pretraining is unknown.

If the first turn returned a plain tactic list, the offline replay can test every prefix of that list without sending another inference request. Successful training plans produce one additional script-SFT row per root with a derived script prompt. Report this post-hoc replay independently of the single-action live score.

```bash
python -m datagoblin.training_gym.rlt_gym.lean_goose_replay \
  --lean /path/to/lean-4.12.0/bin/lean \
  --archive runs/inky-lean-open --out runs/inky-lean-plan-replay
```

The separate List comparison uses the four registered HARD-02F OOD roots, two checks per arm, and a five-candidate native oracle for each root. It writes evaluation receipts only. Prepare Mathlib at `809c3fb3b5c8f5d7dace56e200b426187516535a` with its Lean 4.12 cache, then run:

```bash
python -m datagoblin.training_gym.rlt_gym.lean_goose_list \
  --lake /path/to/lean-4.12.0/bin/lake --mathlib /path/to/mathlib4 \
  --worker apps/lean-goose/target/release/lean-goose \
  --port 8097 --model inky-mini-varloop --out runs/inky-lean-list
```

The native List baselines retain the original fixed order, root-seeded random order, and suffix-length rule. The two-check fixed score and exhaustive kernel oracle answer different questions. Treat solver results on these generated menus separately from open Lean proofs and hard Mathlib or LeanGym performance. The paired pilot uses no sealed confirmation data.
