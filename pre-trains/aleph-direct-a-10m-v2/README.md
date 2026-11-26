# Aleph direct A: first 10M positions (replacement v2 corpus)

The 999,352,320-parameter direct-A model has been trained for exactly 10,000,000 positions (39 update groups) on one A100-SXM4-40GB under the frozen recipe, and the final checkpoint has been converted to safetensors and GGUF. This is an engineering pilot: it shows that the runner, the data cursor and the checkpoint/restore path work at full size. It is not a usable language model, and its final loss of about 7.1 nats/token says nothing about final quality.

## Result

| Item | Value |
| --- | --- |
| Positions | 10,000,000 (39 groups of 262,144; group 39 is the 38,528-position remainder) |
| Supervised labels | 4,825,552 (general 3,494,982; custom research 481,255; verified 496,906; procedural 317,244; failure-directed 35,165), equal to the release receipt |
| Loss, group 1 / 20 / 29 / 38 | 11.2807 / 8.1331 / 7.4763 / 7.0351 (full log: [training_log.csv](training_log.csv)) |
| Gradient norm, group 1 / 38 | 15.899 / 2.072 |
| Learning rate | linear warm-up of the frozen schedule, 1.25e-05 at the end (peak 2.5e-4 is reached at 20M tokens) |
| Throughput | 57.8 to 58.2 s per full group, about 4,520 useful positions/s |
| Peak memory | 20.4 GB allocated, 21.5 GB reserved, of 42.4 GB |
| Hardware | one A100-SXM4-40GB, torch 2.14.1+cu130, explicit BF16 compute with FP32 master weights and FP32 AdamW moments |
| Corpus | replacement v2, archive SHA-256 `a0579b316349cb11ede4e3833cdfcb9ec852755f0442ff68dfffef579e4ccdfe`, index `53d27abfcea62a06dccf4e2463c7b83b9b3693abfca4f796d24536e5282730c0` |

The original Aleph-10M archive was lost, so this is a new train-only release with the same recipe. Do not compare its losses with the original private evaluation panel.

## Restart parity at full size

The run was done in four Colab sessions with checkpoints on Google Drive. Groups 20 to 24 were run twice, once uninterrupted and once after restoring the group-19 checkpoint on a fresh VM, and the two runs agree exactly: loss and gradient norm are identical to every printed digit and the per-lane label counts are identical. Group 29 reproduced the same way after a second restore. The restore covers weights, AdamW moments, RNG state and the data cursor together.

## Export

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| native/model.safetensors (FP32, 14 unique blocks) | 3,997,421,432 | `2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8` |
| llama24/model.safetensors (BF16, 24-layer Llama layout) | 3,347,280,432 | `f39e58d61eb380347551b410e53b4990c140cc9a24227ac28362586114a6a681` |
| gguf/aleph-direct-a-10m-v2-bf16.gguf | 3,349,237,888 | `d7b9290f7fe21bd500025365b5b2d3d5bfc554ac5b87d2581b95629c6271cdc2` |
| gguf/aleph-direct-a-10m-v2-Q4_K_M.gguf | 1,029,509,248 | `c66b3756710fe71ab4501abdfadcfc3ce0b7e2a3acb2c7e61b1991acc2b40d5f` |

The files are on Google Drive under `MyDrive/aleph-v2-run/export/` with a `MANIFEST.json` of every hash, next to the training checkpoints (`group_001`, `group_019`, `group_029`, `group_039_7fbdb1f91cf8`). A private repository `AlephFunk/aleph-direct-a-10m-v2` exists on Hugging Face and is empty; the upload has not been done.

The model uses 4 shell/answer blocks (FFN 8192) and 10 shared core blocks (FFN 8960), run in the order [0, 1..10, 1..10, 11, 12, 13]. The Llama-layout export writes the shared blocks twice, zero-pads the narrow FFNs to 8960 (function-preserving) and reorders the q/k projections from interleaved to half-split rotary layout. It is 1.67 B stored parameters, which is why Q4_K_M is 1.03 GB.

## Checks on the export

- Native FP32 against the Llama-layout BF16 model on a 24-token sequence: mean absolute logit difference 0.0095 on logits of mean magnitude 0.80 (1.2% relative), argmax agreement 23 of 24 positions. This is a BF16-rounding comparison, not an FP32-exact one (the FP32 version ran out of memory on a 12 GB runtime and was not repeated).
- llama.cpp tokenizes four test prompts to exactly the ids of the original `tokenizers` library (SmolLM2 tokenizer; its SHA-256 matches the recipe pin `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`).
- The first generated token of both GGUFs equals the native FP32 model's top choice on 4 of 4 prompts. Run the GGUF with `llama-completion -no-cnv`; `llama-cli` applies a chat template this model was not trained for.

## Cost and runs

Four A100 sessions did the training (groups 1-9, 1-24, 20-29 and 30-39; the second and third repeat steps to test restore), one more A100 session trained nothing while Drive access was being sorted out, and a free CPU runtime did the conversion. The account balance went from 999.53 to 974.92 compute units (about 24.6 units) over the whole effort, well above the original 10-unit ceiling, because of timed-out allocations that billed without a usable runtime, one lost run (checkpoints written without Drive) and that idle session. The training itself is 39 groups at about 58 s each. The TPU v6e-1 requests produced no TPU result, and the runner's TPU path was not qualified on hardware.

## Reproduce and receipts

Receipts, logs, the CUDA backend for `colab10m_xla.py` and the launcher are in PR 74, branch `research/aleph-recovery-sources-20261125`, folder `training/aleph_next/recovery/` (`A100_PILOT.md`, `receipts/`). The corpus verifier is `training/aleph_next/recovery/verify_bundle.py`.

## Limits

- One device, one backend (CUDA). TPU and distributed training are not qualified.
- No held-out evaluation was run, and the loss at 10M positions is not a quality estimate.
- No license has been chosen for the weights. The tokenizer is Apache-2.0 and the corpus includes CC BY 4.0 text.
