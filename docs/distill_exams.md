# DataGoblin Distill adapters v0.1

Canonical compiler: [Compactification-Theory Distill v0.1.0](https://github.com/MoralityLabAI/Compactification-Theory/tree/ac6ec9243d16f73117a986307851cd031bca7fa8/Distill), pinned at ac6ec9243d16f73117a986307851cd031bca7fa8. These sidecars add independent design exams; existing runtimes, frozen data, and verifier boundaries are unchanged. Use only reviewed public development inputs. No model calls, training exports, or automatic ingestion.

- RLT-ROLLOUT: `src/datagoblin/training_gym/rlt_gym/distill_exam.json`
- DG-AGN-TINKER-01: `Agonist/distill_tinker_exam.json`
- MOLT-GOBLIN-01: `src/datagoblin/training_gym/distill_molt_exam.json`
- RLT-POSTFAILURE: `src/datagoblin/training_gym/rlt_gym/distill_postfailure_exam.json`

From the pinned Compactification-Theory checkout, run `python -m Distill compile --adapter /absolute/path/to/adapter.json --out /tmp/exam.json`. Use `python -m Distill prompt --packet /tmp/exam.json` to render the examination. Use the existing native verifier independently; narrative fields require review. These adapters describe mainline design sources, not unmerged successor experiments.
