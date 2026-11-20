# Archive and provenance

The original artifacts were recovered from the persistent user Library on 2026-11-19. Their hashes were computed over the original bytes. GitHub stores full extracted PDF text under `training/reports/`; the byte-identical original PDFs remain in Library, and are linked here.

| Artifact | SHA-256 | Source |
| --- | --- | --- |
| Organelle Gym Research Design, 13-page PDF | `e921b5be770e8f6171925f571ae64ce35f62c443c48cf54a90700e8bc1a0bcf1` | [Original PDF in Library](https://chatgpt.com/library?file_id=libfile_fde3_f-6a204ebcd75c8191b05833f33fb2b384_c26eca361130819193bd5d8e21b255a7) |
| Aleph 1B Research Dossier, 32-page PDF | `5aa18ac872ef3d856cb5c898b6b6d49c6f8c4c5b799e0d4da505afb25db9ce91` | [Original PDF in Library](https://chatgpt.com/library?file_id=libfile_fde3_f-d241b573d780819190cc95e761a1d221_c26eca361130819193bd5d8e21b255a7) |
| Aleph research source ZIP | `3d204e36cff354e1afb1bc42673405b0fd8994e0fbdd01cad4ea285196e3567a` | [Original ZIP in Library](https://chatgpt.com/library?file_id=libfile_fde3_f-b7dcda0c695081918c86fd25e09249b6_c26eca361130819193bd5d8e21b255a7) |

### Source ZIP inventory and checksums

The 2026-11-19 source bundle contains an earlier copy of the PDF (different PDF SHA), an annotated 48-paper catalog, source-cited HRM audit, JSON experiment specification and a standalone cost calculator. The exact original files remain in the ZIP.

| Path inside bundle | SHA-256 |
| --- | --- |
| `aleph_compute.py` | `c7cfc5333e3458e50c2b1192d23c4450bba5d67b2dc7cf5122105740beafd553` |
| `aleph_compute.json` | `95afed47d50f981b7e79906f11e0e3c922355eb67b8d11b5c796e886e4b5229c` |
| `aleph_experiment_spec.json` | `50c1fd8827ccb6a3746ef01d3ad6392e59b82486cad9bd6dc4c2c767626533eb` |
| `hrm_evidence_audit.json` | `d7c4f9081e57b8028f35002f5bcb30a68f69e365aedb16fc9f1eb1d4f3a961c3` |
| `looped_transformer_catalog.json` | `be300e1f7f2306ac4e422bc65fd7ba0ee54d673bbb6ae13edf8b5241d37ebf77` |
| `looped_transformer_catalog.bib` | `2e5a5fb074a069fbaa8a3fa1154aa8de66ea142d1c48f50536182b32227ced9e` |

### Key code and result pointers at design time

- `MoralityLabAI/DataGoblin` starting tree: `02e9918dc7bb23a06bfb3f648593a8fff9719b7d`; existing architecture gym is mock-stage only.
- [BPD DT04 results](https://github.com/MoralityLabAI/BPD/blob/1082496445cb99e85cf346597aa87d4cb28f7059/mechinterp/bpd/experiments/depth_steering_gym/dt04/RESULTS_20261119.md): 59 roots; static rank-8 state-space edits failed their mechanistic advantage gate, while operator-space fits showed descriptive extrapolation in the competent subset.
- [Deuterium results](https://github.com/MoralityLabAI/LoopedTransformers/blob/02890243fba1e39764b53928bfb316dd501b530a/mechinterp/deuterium/README.md): development no-registered-advantage result, pass-mask caveat and unresolved prospective numeric gate.
- [Sapient HRM-Text repository](https://github.com/sapientinc/HRM-Text), [HRM-Text-1B checkpoint](https://huggingface.co/sapientinc/HRM-Text-1B), and [original language-model paper](https://arxiv.org/abs/2605.20613).

The generated Markdown transcriptions and new dyad note are maintained here for searchable Git history. The authoritative original report layout and the full structured bibliography are retained at the original Library links.
