# Replacement 10M v2: verified build

CI [45722542334](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45722542334/job/139584146676) completed generation, detached source replay, 39-group stream replay, restart replay and a second fresh-extraction replay. No accelerator used.

[Download the actual ZIP and release receipt](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45722542334/artifacts/15073685467). The downloaded Actions archive contains Aleph-10M-replacement-v2.zip and RELEASE.json. Extract that outer archive first. The inner file is 34,682,718 bytes.

Archive SHA-256: a0579b316349cb11ede4e3833cdfcb9ec852755f0442ff68dfffef579e4ccdfe

Index SHA-256: 53d27abfcea62a06dccf4e2463c7b83b9b3693abfca4f796d24536e5282730c0

Replay SHA-256: e274e031922e48ef40808c058bd20d62233648117771966633620b3a9ad133bc

Release JSON SHA-256: ce2d0d5c637b39945fed7d92f022da3d92ab575d4b0e79fb35b8553a8dfaf509

The release JSON from the verified build is also committed as RELEASE.v2.json. The original 10M archive remains missing. This is a new train-only release, with the same recipe, new root ordering and evaluation custody limits recorded in ADMISSION.json. Do not compare its future training quality with the original private panel.

## Verify before allocating a TPU

From PR 74 checkout, install the pinned tokenizer:

    python -m pip install -e . tokenizers==0.22.2
    python training/aleph_next/recovery/verify_bundle.py --archive Aleph-10M-replacement-v2.zip --release training/aleph_next/recovery/RELEASE.v2.json --release-sha256 ce2d0d5c637b39945fed7d92f022da3d92ab575d4b0e79fb35b8553a8dfaf509 --out runs/aleph-v2-verified

The command checks all bytes, re-tokenizes and recomputes generated labels, verifies archived Lean receipts and exactly replays the 39 batches and restart. It runs on CPU. Keep the output directory fresh. A separate Actions job verifies the exact pinned artifact and, if allowed by repository rules, copies it to the evidence branch as assets/evidence/Aleph-10M-replacement-v2-a0579b31.zip and the corresponding RELEASE.json. No other archive may overwrite those names.

## Bounded TPU pilot

Only after the above passes, transfer the verified directory and committed release receipt to the target runtime. Install matching torch and torch_xla. Use the original TPU fixture first. No production launch, no cross-corpus resume. Budget the external runtime, including setup and checkpoint copies, within the remaining original authorization. The launcher requires the release receipt and exact hash:

    python -u training/aleph_next/recovery/train_replacement_xla.py --execute --base runs/aleph-v2-verified --release training/aleph_next/recovery/RELEASE.v2.json --release-sha256 ce2d0d5c637b39945fed7d92f022da3d92ab575d4b0e79fb35b8553a8dfaf509 --out /path/to/persistent/new-output --max-groups 1 --wall-seconds 1200 --checkpoint-reserve-seconds 180

The wall values are examples only; derive actual limits from the remaining compute units and measured billing rate. Stop and disconnect the runtime after the pilot. Full-size hardware fit and actual optimizer training are unverified.
