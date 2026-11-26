# COG-13C native Lean archive import

Source: [HybridTRMLDT PR #37](https://github.com/MoralityLabAI/HybridTRMLDT/pull/37), commit a50e53f2a527d8e9464538dbc0ae2b2603c90bee; [successful native run 45852214025](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45852214025), [artifact 15136527625](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45852214025/artifacts/15136527625).

The importer requires the exact ZIP with SHA-256 d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea. The archive contains generated Mathlib-backed exercises, not original Cogentic paper traces. It is an external native evidence adapter; it does not change the existing RLT corpus builders or train a model.

## Reproduce

Download the original Actions artifact ZIP (authenticated GitHub access may be required; the artifact has limited retention). From the repository root, with Python 3.10+:

```bash
python training/cogentic_native/native_cog13c_ingest.py /path/to/cog13c-native-green.zip --out /tmp/cog13c-datagoblin
```

Use a new output directory. The importer never extracts arbitrary ZIP paths. It validates the pinned archive, exported file hashes, 24 distinct task statements and split custody; binds native mailbox requests and responses; independently decodes dense, rank-four float32 and rank-four 4-bit packets and checks recurrent scores; verifies the certificate source and compiler log hashes, independent compilation receipts, allowed axiom sets and in-budget credit. Training labels and residual-call targets are checked against the recorded trajectories. It does not invoke Lean anew or independently reconstruct neural training, sender feature vectors, or every scheduling decision.

## Executed audit

The exact source in this branch (Git blob d816ae9e77f55221f4dd3ef739a0a0dec646f8f9) was executed twice against the actual native archive. The second run checked exact pre-action teacher goals and recomputed allocator features. Both exports have identical hashes. A tampered archive was rejected.

| Audit metric | Count |
|---|---:|
| Root tasks | 24 |
| Training teacher probes | 224 |
| Portfolio probes (training and readout) | 822 |
| Decoded packets | 493 |
| Certificate compiler processes | 128 |
| Certified episodes across arms, including training | 64 |
| Tactic outcome examples, train only | 224 |
| Allocator residual examples, train only | 44 |

The ordinary and deeper evaluation roots are excluded from both training streams. Tactic outcomes encode printed-goal change/closure, not guaranteed usefulness or fully certified theorem progress. The allocator labels are remaining probe calls from eight certified TRAIN trajectories, not CPU seconds or memory bytes.

Output files:
- tactic_outcomes.jsonl: SHA-256 8207309de2f134aea952bd2a4014861853bc839b1cc54074236207f34794ffe9
- allocator_residuals.jsonl: SHA-256 4a52879b8cee47446d472d28b6346a7599f3049dcca944e753a36997fc5d2845
- manifest.json: source, hashes, full independent audit and per-arm evaluation summaries.

Both streams carry root lineage, source artifact identity and response-only target metadata. Root identifiers and provenance are outside the model input. The 268 rows are a small development pilot; no generic training claim, neural LDT fit, or downstream SFT run follows from this import. Generated corpus files remain outside the source tree. A future adapter can mix these outcomes with other train-only verifier traces after explicit split/lineage checks.
