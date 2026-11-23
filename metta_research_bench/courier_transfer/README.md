# DG-METTA-12 | Storyworld courier action-set transfer

**Status: CPU reference wiring (2026-11-23).** The reported native Storyworld courier commit `4cfe171` could not be resolved on the remote `MoralityLabAI/metta-storyworld` repository; its main tree currently has no courier kernel. The local three-test native report and uncommitted DataGoblin comparison could not be audited here. Native transfer remains gated.

## Preregistered question

Does teaching the Looped Transformer to assign probability to **all equally good actions** improve verified completion when the actions are courier navigation, evidence collection, permit acquisition, pickup, delivery, and STOP?

| Arm | Training objective | Inference |
| --- | --- | --- |
| A | Lexicographically first action among all exact optima | Looped Transformer |
| B | SHA-256 hashed singleton from the exact same optimal set | Looped Transformer |
| C | Full optimal-action-set log-sum-exp objective | Looped Transformer |
| D | Exactly C's frozen checkpoint | Transformer plus certified MeTTa STOP macro |
| E | No training | Exact bounded oracle reference |

The **primary confirmation** is C minus B on full verified deliveries followed by a correct STOP. A is the additional canonical-label control. D minus C is the separate inference comparison. Prior MeTTa curation and docket gains motivate this test and are not courier outcomes.

## Split firewall

`roots_manifest.json` freezes 520 deterministic root IDs, root-seed derivation, and a SHA-256 fingerprint of the complete ordered ID list. Train/development/frozen use distinct roots from four families; transfer uses withheld family names.

| Split | Families | Roots each |
| --- | --- | ---: |
| Train | route_ties, permit_gate, multi_drop, evidence_ledger | 48 |
| Development | same four | 12 |
| Frozen primary | same four with fresh roots | 40 |
| Transfer | permit_chain, evidence_collision, unseen_topology | 40 |

For each seed 1101, 1102, 1103, use the same initial weights, public state features, valid-action vocabulary, state presentations, batch indices, 600 AdamW updates and 48 examples per update. Choose B by the documented stable hash. C uses `-log(sum_{a in A*(s,h)} softmax(z)[a])`. Count one training presentation per state regardless of tie multiplicity.

A root qualifies for successful-action training only when the exact bounded oracle can finish all deliveries. Root-local and seed-local effects are preserved. Frozen primary is held closed until the source, data and checkpoint SHA-256 receipts and the development report are locked. Paired 95% root-clustered bootstrap retains all model seeds within each resampled root. The prespecified C-B success gate is an interval lower bound above zero. Always report every seed and transfer result, including failures.

## Gates and reproduction

1. Run `python -m unittest discover -s tests/unit -p test_dg_metta12_reference.py -v` and `python metta_research_bench/courier_transfer/manifest_check.py` on CPU. This independently implemented courier model only establishes a wiring gate.
2. Push the complete native courier commit to Storyworld, pin its full SHA and runtime, run the three original targeted tests, then implement the adapter specified in `ORACLE_CONTRACT.md`. Differentially replay small shared states against the Python reference and inspect all disagreements.
3. Train A/B/C using native verified oracle action sets; pin the upstream LoopedDecoderLM revision stated in `protocol.json`. Lock source and checkpoints before opening frozen roots.
4. Evaluate D with a native two-check atomic macro only when C's model proposes STOP and at least two checks remain. Compare C and D with the same five-call ceiling, reporting actual calls, calls per success, vetoes, false stops and paired regressions. Abort if no native rollback/macro transition exists.
5. Emit one `RESULT_SCHEMA.json`-valid receipt per root, arm and seed. Distinguish grading checks from budgeted policy checks, persist proof identities, and label any Python-only record `reference_wiring`.

A green wiring workflow cannot be reported as native-domain transfer or as a trained Looped Transformer result.

Files: `protocol.json` preregisters all arms and metrics; `roots_manifest.json` freezes the cohort allocation; `ORACLE_CONTRACT.md` defines admissibility, Bellman ties and transaction semantics; `RESULT_SCHEMA.json` defines receipts; `reference.py` and the unit tests are CPU controls.
