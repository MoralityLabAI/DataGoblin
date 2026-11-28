# DataGoblin training research archive

**Status, 2026-11-28:** active 10B continuation plan; the 10M direct-A engineering pilot completed. See the [10B Colab handoff](reports/2026-11-28_aleph_10b_colab_handoff.md) and [pilot receipt](../pre-trains/aleph-direct-a-10m-v2/README.md). No additional compute spend is authorized here.

This directory records the proposed **Data Goblin Aleph 1B** architecture and an optional **HRM-Text-1B dyadic companion**. The aim is a local model that creates and evaluates research gyms, curates data, predicts mechanistic interventions and learns research judgment from independently verified outcomes.

## Reports

- [Active 10B curriculum](data/aleph_10b_mix.json) and [Colab continuation handoff](reports/2026-11-28_aleph_10b_colab_handoff.md). First new stage: 10M to 100M; preserve the native optimizer checkpoint.
- [Data landscape and 20B / 30B token plans](reports/2026-11-19_data_landscape.md). Historical scaling alternatives, conditional on the 10B endpoint and measured throughput, quality and budget.
- [Data catalogs and proposed mixture](data/README.md). Public sources, owned corpus receipts, repeat limits and admission requirements.
- [Organelle Gym Research Design (13-page PDF text)](reports/2026-11-19_organelle_gym_pdf_text.md). Source PDF SHA-256: `e921b5be770e8f6171925f571ae64ce35f62c443c48cf54a90700e8bc1a0bcf1`.
- [Aleph 1B Research Dossier (32-page PDF text)](reports/2026-11-19_aleph_1b_dossier_pdf_text.md). Source PDF SHA-256: `5aa18ac872ef3d856cb5c898b6b6d49c6f8c4c5b799e0d4da505afb25db9ce91`.
- [Dyadic HRM companion proposal](hrm_text_dyadic_companion.md). New design note, proposed and untested.
- [Compute assumptions and 700 H100-hour envelope](estimates/README.md).
- [Runnable analytical compute model](estimates/aleph_compute.py).
- [Experiment protocol and gates](aleph_protocol.md).
- [MeTTa source catalog and curation gym](../MeTTa/README.md). Pinned language source inventory, original MeTTa advisory examples, offline policy fixtures, and split-safe generated candidates.\n- [RLT gym collection for the compact learner](rlt_gym/README.md). Executable CPU controls, teacher repair queues, and checked training exports.
- [Source artifact manifest](SOURCES.md).

The existing project machinery is in [`docs/training_gym.md`](../docs/training_gym.md), [`docs/hrm_text_milestone.md`](../docs/hrm_text_milestone.md), and [`docs/model_training_roadmap.md`](../docs/model_training_roadmap.md). The training gym includes offline development scaffolding; the separate direct-A receipt establishes full-size engineering execution, not model-quality promotion.

The text reports are searchable transcriptions of the archived PDFs. They preserve page sequence, while PDF table layout can flatten. The original byte-identical PDFs and the original ZIP bundle remain separately preserved in the ChatGPT Library; see `SOURCES.md`.

## Scope and boundaries

- **Primary model:** direct-A, 999,352,320 unique parameters, with a completed 10M engineering pilot. The initial endpoint is 10B cumulative processed non-padding positions; the earlier four-operator design and 20B/30B extensions remain alternatives.
- **Scientific hypothesis:** causal, independently checked feedback about messages between recurrent modules improves held-out task composition over a capacity-matched task-trained router.
- **Companion:** an independently loaded `sapientinc/HRM-Text-1B` checkpoint, invoked on selected reasoning decisions. The dyad is a two-model system, not a claim that both fit in 1B parameters.
- **Compute:** 700 H100 GPU-hours is a scenario, not an established Silico entitlement or billing conversion. The 2026-11-19 reported remainder was $1,560 equivalent in Silico RUs.
- **Reproducibility:** pin pretrained model revision, tokenizer, data source/license, curriculum lineage, hardware, optimizer state, actual billed occupancy and scorer semantics before a charged run.
- **Research authority:** proposals, VPD/BPD attribution and model votes remain hypotheses. Environment-owned verifiers produce the authoritative task measurements. Keep sealed evaluation splits out of the training and selection path.

Training, downloads, paid jobs, and checkpoint edits require a separately declared capped runner and explicit authorization. The files in this folder are design and estimate artifacts.

## Reading the archived plans

The original PDF transcriptions preserve earlier alternatives, including smaller runs and a 10B flagship endpoint. The historical [data report](reports/2026-11-19_data_landscape.md) evaluates the user's later 20B/30B direction. Its quotas are proposals whose admitted source capacity and provider billing still need measurement. The earlier reports remain available as the design history.
