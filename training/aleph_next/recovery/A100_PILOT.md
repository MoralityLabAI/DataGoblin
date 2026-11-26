# Direct A on one A100: replacement-v2 pilot, groups 1-9

Run on 2026-11-26 on a Colab A100-SXM4-40GB (42,405,855,232 bytes), torch 2.14.1+cu130, CUDA 13.0, one device, no TPU. `colab10m_xla.py` gained a `cuda` backend and `train_replacement_xla.py` gained `--device xla|cuda`; the explicit-BF16, FP32-master, FP32 AdamW math is unchanged. The corpus is the pinned replacement v2 (archive `a0579b31...ccdfe`, index `53d27abf...730c0`, release JSON `ce2d0d5c...f8dfaf509`), replayed on CPU before provisioning. Recipe, architecture, tokenizer, optimizer and lane mixture are frozen.

## What ran

| Step | Result |
| --- | --- |
| CPU FP32 and BF16 fixtures | passed (FP32 gradient relative L2 4.8e-7; BF16 8.8e-3) |
| CUDA BF16 fixture on the A100 | passed: gradient relative L2 8.89e-3 against a 0.12 limit, document isolation, all-masked pack, unequal label counts, fixture restart |
| Group 1 from scratch | 262,144 positions, 137,391 labels, finite update, checkpoint written in 121.2 s |
| Resume from the group-1 checkpoint | hashes and run identity verified, weights, AdamW moments, RNG and lane cursor restored, groups 2-9 continued in the same process |
| Stop | wall cap discarded the incomplete group 10; the step-9 checkpoint was written in 122.6 s |

Stopped at 2,359,296 of 10,000,000 positions (9 of 39 groups) because of the 10 compute-unit ceiling. Checkpoints were written to the runtime's local disk. Drive was not mounted, so no checkpoint outlived the runtime. Only the JSON receipts and logs survive; they are in `receipts/`.

## Numbers (per update group of 262,144 positions)

| Group | Loss | Gradient norm | Learning rate | Wall s | Positions/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.2807 | 15.899 | 3.277e-07 | 58.1 | 4514 |
| 2 | 11.2388 | 12.433 | 6.554e-07 | 58.2 | 4507 |
| 3 | 11.1611 | 12.976 | 9.830e-07 | 58.0 | 4517 |
| 4 | 11.0512 | 12.559 | 1.311e-06 | 57.8 | 4538 |
| 5 | 10.8879 | 12.352 | 1.638e-06 | 57.9 | 4528 |
| 6 | 10.7059 | 11.680 | 1.966e-06 | 58.0 | 4520 |
| 7 | 10.5295 | 10.670 | 2.294e-06 | 57.8 | 4537 |
| 8 | 10.2856 | 9.983 | 2.621e-06 | 57.9 | 4527 |
| 9 | 10.1252 | 9.349 | 2.949e-06 | 58.0 | 4518 |

Peak allocated memory was 20,388,972,032 bytes and peak reserved 21,483,225,088 bytes of 42,405,855,232. Supervised labels over the nine groups were 1,145,561: general 824,209, custom research 115,294, verified 116,262, procedural 81,422, failure-directed 8,374.

## Cost

The A100 billed 8.16 compute units per hour. Total consumption for the task was 9.04 units (balance 999.53 to 990.49), including 1.94 units charged by two timed-out TPU v6e-1 requests that returned no runtime. At 58 s per group, all 39 groups need about 38 minutes of training plus two or three checkpoint saves of about 2 minutes each, roughly 5.5 compute units from a warm start.

## Limits

- The resume was checked for hash, identity, cursor and loss continuity (group 2 loss 11.2388 follows group 1 at 11.2807). It was not compared to an uninterrupted run, so bit-exact restart parity at full size is not demonstrated.
- TPU v6e-1 was requested three times; Colab answered with two read timeouts and one `Service Unavailable`. No TPU result exists.
- The nine-group loss path covers 23.6% of the first 10M positions with a warm-up learning rate, so it says nothing about final quality.
- The corpus is the replacement v2 release, not the original Aleph-10M archive. Its results are not comparable with the original private evaluation panel.

## Rerun with Drive persistence (2026-11-26, second A100 session)

Same runner and corpus, fresh A100-SXM4-40GB (8.16 units/hr). Drive was mounted through the CLI's own flow before the checkpoints were due; a sync process on the VM copied each finished checkpoint into `MyDrive/aleph-v2-run/`, compared every file by SHA-256, and wrote the pointer last. The Drive connector independently lists `group_001_0454b5ce16ea` and `group_019_3f3c570e1824` there, with `CHECKPOINT_001.json`, `CHECKPOINT_019.json` and `PROGRESS.json`.

| Item | Result |
| --- | --- |
| Groups trained | 24 of 39 (6,291,456 of 10,000,000 positions), one uninterrupted process |
| Loss, group 1 to 24 | 11.2807 to 7.8077 |
| Gradient norm, group 1 to 24 | 15.899 to 3.195 |
| Time per group | 57.8 to 58.3 s, 4,498 to 4,538 useful positions/s |
| Peak memory | 20,390,020,608 bytes allocated, 21,483,225,088 reserved of 42,405,855,232 |
| Supervised labels, 24 groups | 3,038,681 (general 2,198,484; custom research 304,873; verified 311,585; procedural 201,596; failure-directed 22,143) |
| Checkpoint saves | group 1 and group 19 each took about 121 s on the VM and 146 s to copy and verify on Drive |
| Last durable checkpoint | group 19, 4,980,736 positions (loss 8.2788), verified on Drive |

The group-24 checkpoint was in its `.partial` write at 03:52:26Z and does not exist on Drive. The external stop at 03:58:30Z ended the runtime before the sync published it; whether the save or the copy was unfinished is not known because the VM's logs were lost with it. Groups 20 to 24 are therefore recorded in `records_groups_1_24.json` but cannot be resumed from. Resuming needs a new runtime, a Drive mount, and the group-19 checkpoint.

The first lost run (no Drive) and this one show the same step costs. Cost for this session: the runtime billed 8.16 units/hr from about 03:14:50Z to 03:58:30Z (about 5.9 units). The balance readout lagged the stop time, so the exact figure should be read from the account.
