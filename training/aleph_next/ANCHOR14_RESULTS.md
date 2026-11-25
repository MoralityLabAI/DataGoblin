# 14M competence qualification

The direct reference cleared both native developer anchors on all three seeds after 128 continuation updates. All three fixed-damping candidates then cleared the same anchors at the locked horizon. The first registered horizon, 64, yielded zero qualifying direct seeds.

These are three seed repeats on the original 4,096 developer byte positions and the same 96 native repair examples. Text loss is in natural-log units per next byte. Repair reports correct answers out of 96.

| Seed | Direct 64 NLL | Direct 64 repair | Direct 128 NLL | Direct 128 repair | Fixed damping 128 NLL | Fixed damping 128 repair |
|---:|---:|---:|---:|---:|---:|---:|
| 137 | 3.260303 | 42 | 3.101485 | 74 | 3.118316 | 71 |
| 163 | 3.251776 | 45 | 3.101031 | 77 | 3.127801 | 85 |
| 191 | 3.294014 | 32 | 3.175343 | 74 | 3.113346 | 64 |

The registered native text threshold was the train-fitted add-one unigram at 3.235671 nats/byte. Four-way chance is 25%. Copying the final operand gets 59/96 repair items and sets the stronger registered repair threshold. At horizon 128, direct solved all 47 SET items for every seed, plus 27, 30 and 27 of 49 ADD items. Damping solved 43, 46 and 47 SET items, plus 28, 39 and 17 ADD items.

The train-only interpolated byte trigram reaches 2.410623 nats/byte on these windows, using eight pseudo-observations per level and resetting context at each window. A majority lookup trained on the explicitly repeated public previous-state, operation and operand covers all 32 transition keys and solves 96/96 repair items. A lookup on operation and operand alone solves 59/96. Both networks clear the preregistered minimum bars. The stronger count models retain a substantial advantage on these original tasks. The independent confirmation tests lengthened arithmetic compositions, changed graph structures and new documents.

Mean developer NLL is 3.125953 for direct and 3.119821 for damping. Damping wins this endpoint on seed 191. Mean repair accuracy is 78.125% for direct and 76.389% for damping. Damping wins this endpoint on seed 163. The independent confirmation panel supplies the placement comparison.

## Matched exposure and auditing

Both arms started at their own verified 32-update native prefix. Each has 14,179,584 physical trainable parameters and 24 logical block visits. Both used the same architecture, optimizer restart, learning-rate schedule, microbatch order and targets. The 23 nonterminal local coefficients for fixed damping are each 0.120693833815. The terminal operation uses the native block result exactly.

| Seed | Prefix inputs | Continuation inputs | Continuation padded positions | Total processed real inputs | Total padded positions |
|---:|---:|---:|---:|---:|---:|
| 137 | 11,523 | 40,702 | 53,014 | 52,225 | 69,434 |
| 163 | 11,492 | 40,643 | 52,812 | 52,135 | 69,252 |
| 191 | 11,471 | 40,325 | 53,196 | 51,796 | 69,506 |

Every continuation used 4,096 supervised next bytes and 2,400 verified or graded choice labels. The choice labels break down into 544 full native repairs, 512 transition suffixes, 1,024 public compact repair examples, 160 revisions, 128 Lean candidates and 32 counterexamples. The resulting approximately 0.0037 processed inputs per physical parameter characterizes this low-exposure screen.

All physical parameters received gradients on every update. The recorded preclip norms, sampled every 16 updates, ranged from 1.35 to 5.94 across direct fits and from 1.46 to 36.39 across damping fits. Damping seed 163 reached 36.39 at update 80 and 3.08 at update 96. Global clipping was 1.0 for both placements. Training and evaluation ran under single-core CPU bounds. One direct seed moved from core 0 to core 6 at its first horizon, accounting for a material wall-time difference. Synchronized same-core timing belongs to the separate throughput assessment.

The matched audit recomputed twelve model and twelve training-root hashes across both registered horizons. Every seed pair shared its original prefix stream, continuation stream, supervised counts, real and padded positions, and training-root digest. The direct selection certificate hashes all six terminal receipts. Its SHA-256 is `83796d1dcd2a8db01a66cbd8060ff28cac8c55e24c53caf553ef8f301c935472`. The trainer, registered protocol and corpus code have joint receipt digest `94bdc78d082ed5f430f578178e8a481186aed5e5d2dab1d8440239332ec3b6e6`.

## Reproduction

Use the pinned mixed pack in `runs/placement/mixed` and the six 32-update width-256 checkpoints in `../aleph-next/scale`. The original scale study has the commands that create the prefixes. The following commands assume a fresh study directory at the default path `../aleph-confirm14`. To replay alongside an existing run, pass a fresh `--out` directory to all trainer calls and use that directory's resulting selection certificate. Follow the registered horizon and seed ordering in [the qualification protocol](ANCHOR14_PROTOCOL.md).

```powershell
foreach ($horizon in 64, 128) {
    foreach ($arm in 'direct', 'global_full_damp') {
        foreach ($seed in 137, 163, 191) {
            python training/aleph_next/bounded.py --wall 7200 --core 1 --memory-mib 2048 anchor14_train.py --arm $arm --seed $seed --until $horizon
            if ($LASTEXITCODE -ne 0) { throw "Training failed for $arm $seed at $horizon" }
        }
    }
}
python training/aleph_next/bounded.py --wall 180 --core 1 --memory-mib 2048 anchor14_train.py --select
foreach ($seed in 137, 163, 191) {
    python training/aleph_next/bounded.py --wall 600 --core 1 --memory-mib 2048 anchor14_train.py --arm global_full_damp --seed $seed --score-locked ../aleph-confirm14/selection.json
}
python training/aleph_next/anchor14_controls.py
python -m pytest -q tests/test_anchor14_train.py
```

Each `cells/ARM-SEED/h0128.json` under the study directory records its source and model SHA-256, the corresponding training-root file, full stream checksum, exact token ledger and gradient samples. Adjacent `h0128.pt` files contain plain weight dictionaries. The three `h0128-locked-dev.json` records bind the candidate scores to the same selection certificate. The h0064 files retain the initial failed reference horizon. The control command regenerates `native-controls.json` with SHA-256 `48621d4b9bde4ec1540960af723cd13c7816f78d1181ff4048b26ac761c5ddc8` on the measured host.
