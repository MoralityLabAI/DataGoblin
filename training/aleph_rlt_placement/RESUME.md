# Resume the local placement tournament

Use the existing research checkout and run directory when available. The operator's workspace is `C:\Users\patri\Documents\Codex\2026-11-24\they\work\DataGoblin`. First inspect active workers and timestamps; an absent chat process is not evidence that a fit stopped. The CPU launcher pins training to one allowed processor at low priority with CUDA hidden, an OS memory limit and a wall timeout. On Windows this campaign uses Python 3.12 and CPU PyTorch 2.10.0. Set PYTHONPATH to src or use an editable installation with PyYAML and NumPy. Normal pytest temporary directories should sit outside the checkout for the legacy steward tests.

The archive is a binary member on another branch. Fetch it without changing the research branch. Starting from the repository root:

```sh
git fetch origin evidence/1b-pretrain-v1-20261124
mkdir -p runs/placement
git archive --format=zip --output=runs/placement/evidence-git.zip 48c99a7d9c2d37c82fe127369dfa0124b88bc8ab assets/evidence/DataGoblin-1B-Pretrain-Evidence-v1.zip
python training/aleph_rlt_placement/unpack_evidence.py runs/placement/evidence-git.zip runs/placement/mixed
python training/aleph_rlt_placement/gym.py --smoke
python training/aleph_rlt_placement/cpu_worker.py --wall 180 audit.py --out runs/placement/audit.json
```

The following mixed groups run sequentially. Rerunning an interrupted group with identical arguments verifies existing receipts, skips completed cells and restores partial checkpoints. Source changes to gym.py, study.py or corpus.py invalidate the study config hash. Training and inference are separate; a final checkpoint without a receipt can still need several crossover evaluations.

```sh
D=training/aleph_rlt_placement
python "$D/cpu_worker.py" --wall 10800 study.py --pack runs/placement/mixed --out runs/placement/study --width 80 --steps 192 --r 2 --arms direct --seeds 137 163 191 --roots 48 --probe-roots 6
python "$D/cpu_worker.py" --wall 10800 study.py --pack runs/placement/mixed --out runs/placement/study --width 80 --steps 192 --r 3 --arms direct pass_rlt pass_damp core_edges core_edges_damp --seeds 137 163 191 --roots 48 --probe-roots 6
python "$D/cpu_worker.py" --wall 10800 study.py --pack runs/placement/mixed --out runs/placement/study --width 80 --steps 192 --r 2 --arms pass_rlt core_edges first_last penultimate first_penultimate_last first_penultimate_damp global_full global_full_damp --seeds 137 163 191 --roots 48 --probe-roots 6
python "$D/review.py" --root runs/placement --out runs/placement-review.json
python "$D/promotion.py" --review runs/placement-review.json --out runs/placement/promotion-decision.json
```

The separate original-task diagnostics use width 72, steps 40, batch 8, eval-n 48, threads 1 and the same three seeds in `gym.py`. At r=2 use all twelve ARMS, out `runs/placement/raw2`, seconds 3300; at r=3 use direct, core_edges, core_edges_damp, pass_rlt and pass_damp, out `runs/placement/raw3`, seconds 2400. Add `--resume` to an existing original-task run only. These are loss-on-final-token wiring tasks.

Apply [PROTOCOL.md](PROTOCOL.md) before advancing. Save receipts separately from weights and raw texts. The full-scale ledger and the clock game can run independently:

```sh
python "$D/fullscale.py" --r 3 --candidate pass_rlt
python "$D/clock_order_gym.py"
python "$D/control_audit.py" --help
python "$D/clock_probe.py" --help
```
