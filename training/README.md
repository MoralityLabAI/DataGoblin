# DataGoblin training research archive

**Status:** research design and estimate archive, 2026-11-19. No Aleph 1B model is claimed trained by these documents, and no compute spend is authorized here.

This directory records the proposed **Data Goblin Aleph 1B** architecture and an optional **HRM-Text-1B dyadic companion**. The aim is a local model that creates and evaluates research gyms, curates data, predicts mechanistic interventions and learns research judgment from independently verified outcomes.

## Reports

- [Data landscape and 20B / 30B token plans](reports/2026-11-19_data_landscape.md). Current planning recommendation: 20B tokens with two recurrent rounds, with a conditional 30B continuation after throughput and development measurements.
- [Data catalogs and proposed mixture](data/README.md). Public sources, owned corpus receipts, repeat limits and admission requirements.
- [Five-repository mathematics corpus census](corpus_intake/README.md). Pinned source scanner, source-only Lean declaration inventory, duplicate audit and interim token counts.
- [Semaev-RedTeam development gym](gyms/semaev_redteam/README.md). Six fixed mathematical-claim scope tasks with deterministic scoring, development-only.
- [Organelle Gym Research Design (13-page PDF text)](reports/2026-11-19_organelle_gym_pdf_text.md). Source PDF SHA-256: `e921b5be770e8f6171925f571ae64ce35f62c443c48cf54a90700e8bc1a0bcf1`.
- [Aleph 1B Research Dossier (32-page PDF text)](reports/2026-11-19_aleph_1b_dossier_pdf_text.md). Source PDF SHA-256: `5aa18ac872ef3d856cb5c898b6b6d49c6f8c4c5b799e0d4da505afb25db9ce91`.
- [Dyadic HRM companion proposal](hrm_text_dyadic_companion.md). New design note, proposed and untested.
- [Compute assumptions and 700 H100-hour envelope](estimates/README.md).
- [Runnable analytical compute model](estimates/aleph_compute.py).
- [Experiment protocol and gates](aleph_protocol.md).
- [Source artifact manifest](SOURCES.md).

The existing project machinery is in [`docs/training_gym.md`](../docs/training_gym.md), [`docs/hrm_text_milestone.md`](../docs/hrm_text_milestone.md), and [`docs/model_training_roadmap.md`](../docs/model_training_roadmap.md). The training gym currently runs offline deterministic scaffolding; real architecture promotion and 1B training are unverified.

The text reports are searchable transcriptions of the archived PDFs. They preserve page sequence, while PDF table layout can flatten. The original byte-identical PDFs and the original ZIP bundle remain separately preserved in the ChatGPT Library; see `SOURCES.md`.

## Scope and boundaries

- **Primary model:** approximately 1.007B unique neural weights with learned communication among four recurrent operators. The original 10B endpoint remains an intermediate checkpoint; the latest data study proposes 20B processed non-padding tokens with a conditional 30B continuation. The reference design uses 2 synchronous rounds and explores 3.
- **Scientific hypothesis:** causal, independently checked feedback about messages between recurrent modules improves held-out task composition over a capacity-matched task-trained router.
- **Companion:** an independently loaded `sapientinc/HRM-Text-1B` checkpoint, invoked on selected reasoning decisions. The dyad is a two-model system, not a claim that both fit in 1B parameters.
- **Compute:** 700 H100 GPU-hours is a scenario, not an established Silico entitlement or billing conversion. The 2026-11-19 reported remainder was $1,560 equivalent in Silico RUs.
- **Reproducibility:** pin pretrained model revision, tokenizer, data source/license, curriculum lineage, hardware, optimizer state, actual billed occupancy and scorer semantics before a charged run.
- **Research authority:** proposals, VPD/BPD attribution and model votes remain hypotheses. Environment-owned verifiers produce the authoritative task measurements. Keep sealed evaluation splits out of the training and selection path.

Training, downloads, paid jobs, and checkpoint edits require a separately declared capped runner and explicit authorization. The files in this folder are design and estimate artifacts.

## Reading the archived plans

The original PDF transcriptions preserve earlier alternatives, including smaller runs and a 10B flagship endpoint. The current [data report](reports/2026-11-19_data_landscape.md) evaluates the user's later 20B/30B direction. Its quotas are proposals whose admitted source capacity and provider billing still need measurement. The earlier reports remain available as the design history.
