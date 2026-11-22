# Inky Mini and Goose Lean rollout pilot

The rollout uses Goose GDK `0.1.0-alpha.14`, a single resident llama.cpp b12602 provider, and native Lean 4.12.0. The shared GPU process on port 8097 exited before the first attempt. The live pilot used a CPU process with the same `TML-d24-varloop-research-IQ4_NL.gguf` checkpoint and one model loop. The file SHA-256 was `e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2`. The provider reported IQ4_NL at 4.5 bits per weight. Server startup to ready took 24.11 seconds.

## Live open tactics

The single-action protocol at commit `ca39d4ca95984680b6c72ed8494add545875c4cb` visited six roots. Lean certified one proof, four episodes exhausted their visit budgets, and one ended with a provider error. The runner spent 31 GDK requests and 37 native compiler calls, including initial goal probes and the final proof replay. All 30 completed provider responses agreed with exact tokenizer preflight. Their largest prompt used 225 tokens, with up to 64 more tokens reserved for completion. Output totaled 472 tokens. Median complete request latency was 12.42 seconds and maximum latency was 266.88 seconds. These are whole-request times on the CPU server, which received other traffic during the run. Dollar cost was unmeasured.

The principal failure was repeating an `intro` action after Lean had already opened the binders. Some first replies contained helpful subsequent tactics, but the protocol tried the first action and requested a fresh continuation. The independent replay at commit `89b954898164f6fac659ee062b6a1173c539a21c` took the first admissible tactic list from each original first-turn response. Lean certified three proofs through replay, including one development proof. The replay made zero further model requests. It is a post-hoc result over the recorded first turns.

The live run produced two action-SFT rows for one training lineage. The replay produced two script-SFT rows for two other training lineages. Both use explicit chat messages and bind their targets to the full native proof certificate. Development and unsuccessful traces remain in evaluation receipts. The four resulting rows exercise the collection path and establish proof provenance for these examples. Learning gains and difficult theorem transfer require their own evaluations.

## Paired List menus

The run at commit `274771a0011caafe07a5f69c7756b42b739b2af4` compared the first four registered HARD-02F OOD roots at seed `2026112247`. The registered statements, tactic menus and two-check budgets were shared across arms. The local verifier imported `Init` under Lean 4.12.0. The registered scripts depend on its core List and equality lemmas. The run checked the pinned Mathlib checkout for provenance and recorded the narrower import.

| Arm | Kernel-certified roots | Candidate checks |
| --- | ---: | ---: |
| Inky Mini through Goose | 1/4 | 7 |
| Fixed order | 0/4 | 8 |
| Seeded random | 1/4 | 7 |
| Symbolic suffix-length rule | 3/4 | 3 |
| Exhaustive five-candidate oracle | 3/4 | 20 |

All seven model requests completed with exact prompt-token agreement. The runner replayed all seven chosen candidates and matched every receipt against the independent oracle. There were zero false STOPs, invalid choices or provider errors in the queued run. The median whole-request wall time was 223.12 seconds with queueing on the shared server. Matched llama.cpp logs recorded a median decode rate of 6.52 tokens per second. The two metrics cover different portions of the request. The paired records are evaluation-only.

## Failed attempts and secondary checks

The first CPU attempt used a preflight that omitted Goose's empty system message. The provider reported three more tokens than preflight on all three returned responses. Those attempts failed closed, and the source also exposed a syntax error in the disjunction fixture. The rerun used Goose's own OpenAI request formatter and an explicit equivalent `Or` statement. Earlier receipts remain archived.

The version-matched [OxiLean pilot](../apps/lean-goose/verification/README.md) verified eight of eight held-out Lean 4.32.0-rc1 controls and rejected both proof substitutions. The OxiLean report records the dependency verdicts separately. The Lean 4.12.0 rollout proofs use their native certificates. Sealed confirmation data stayed unopened throughout this pilot.
