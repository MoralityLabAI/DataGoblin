# Rectified feedback in the Hermes Campsite repair TRM

The October revision measured the effect of having a small TRM write a repaired Campsite grid. The original Hermes menu tries the raw proposal, a C-only projection, and a dual projection. It passed 63 of 148 archived greedy Qwen2.5-3B proposals. Appending a five-seed MeTTa-atom TRM ensemble raised the total to 93 of 148, with zero invalid commits. The public verifier owned every commit. [October paper revision](https://github.com/MoralityLabAI/morality-lab-site/blob/9477211c95cda2a689e6b04a8596cd09f8a51104/papers/MeTTa_Scaffolded_Repair_Curricula_for_TRM_Infused_Hermes_Skills__1_.pdf)

This pilot uses the original Campsite prompt, parser, projections, verifier, MeTTa cell features, and grid decoder. It scores every archived proposal and generates 36 new greedy proposals on a size-stratified subset of the paper's test puzzles. Those puzzles are held out from repair training. New generations on known puzzles measure transfer across proposers. A test on newly generated puzzles would be needed to measure transfer across puzzle distributions. [Qwen2.5-0.5B-Instruct Q4_K_M checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/6dd44a1fb35d11b5d1b28902876ce3cc9e882d0e/qwen2.5-0.5b-instruct-q4_k_m.gguf)

The three arms share the paper's 64-wide repair model, three answer cycles, and three latent updates per cycle. Direct feedback uses the shared block's answer endpoint each cycle. RLT interpolates from the current answer state toward that endpoint with weights `[1/3, 1/2, 1]`. Matched damping uses `[5/12, 5/12, 1]`, with the same total interpolation weight before the final cycle. All arms use the original latent-state update. The fit checks exact final-logit equality between the direct adapter and the paper's implementation.

Each arm sees the same 512 archived training proposals from 256 puzzles and receives 320 AdamW updates per seed. Every training target is a nearest legal solution verified by the original verifier. The training puzzles come from the paper's training pools, with one greedy proposal each from Qwen2.5-3B and Bonsai-8B. The pilot uses two independent seeds per arm and scores single models. The paper's ensemble score is a historical reference.

## Results

The replay reproduced the paper's deterministic score of 63/148 on the Qwen2.5-3B archive. The same menu solved 12/36 of the fresh Qwen2.5-0.5B rollouts. The fresh parser found 36 grids, of which 28 had the full rectangular shape. The raw proposals passed on zero cases in each cohort.

| Feedback | Seed | Archive final | Archive first valid | Fresh final | Fresh first valid |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct | 117 | 72/148 | 79/148 | 16/36 | 17/36 |
| RLT | 117 | 73/148 | 81/148 | 16/36 | 17/36 |
| Matched damping | 117 | 73/148 | 80/148 | 16/36 | 17/36 |
| Direct | 163 | 77/148 | 79/148 | 14/36 | 15/36 |
| RLT | 163 | 78/148 | 81/148 | 15/36 | 15/36 |
| Matched damping | 163 | 78/148 | 81/148 | 15/36 | 15/36 |

The learned repair added 2–4 verified grids on the fresh cohort. RLT added one final-cycle solve over direct feedback in one of two seeds. On cases still open after the menu, RLT and matched damping solved identical final-cycle cases in both cohorts for both seeds. On the archive, RLT's first-valid score was one higher than damping in seed 117 and tied in seed 163. Every commit passed the original verifier.

On fresh cases still open after the menu, RLT and damping reduced mean final answer-state RMS by about 62% relative to their paired direct fits. In seed 117 the RMS was 148.1 for direct, 55.6 for RLT, and 55.7 for damping. In seed 163 it was 126.7, 47.5, and 47.5. This pilot supports a state-scale effect shared by both interpolated updates. It provides little evidence of an advantage from RLT's specific schedule.

On the fresh cohort the final-cycle policy used 112 verifier calls and 288 shared-block calls per fit. Checking for the first valid grid at every cycle used 153–158 verifier calls and 260–280 block calls, depending on the fit. The call counts show the trade-off between added verification and earlier exits. They do not measure a wall-clock speedup.

All six completed fits used single-thread CPU PyTorch after CUDA hit a driver allocation failure. The completed fit stage peaked at 1,025 MiB of Windows job commit. The Qwen replay peaked at 231 MiB of job commit across attempts. A resumed rollout increased the loopback response timeout from 45 to 120 seconds with the same checkpoint, prompts, and decoding. The local evidence digest is `2c4da9edaa4dbf41a9ec73d3c46df76508fac8b26ae1810b916577f7684a3255`.

## Replay

Use the read-only October Hermes study and its Hermes Lite verifier checkout. The input receipts record the source digests. The run needs the PyTorch and RLT pilot optional dependencies, a Windows Job Object wrapper from the Lean CGRT project, and a caller-provided local checkpoint. The CPU rollout supports the official llama.cpp Windows build and runs a hidden loopback-only process inside the same Job Object.

```powershell
$common = @('--out', $out, '--study', $hermesStudy, '--hermes-lite', $hermesLite,
            '--ramcap-package', $leanCgrt)
python scripts/run_hermes_rlt_repair_windows.py prepare @common
python scripts/run_hermes_rlt_repair_windows.py rollout @common --model $qwenGguf --llama-server $llamaServer
python scripts/run_hermes_rlt_repair_windows.py fit @common --cpu
```

The experiment stores proposal digests, per-puzzle verifier results, per-cycle answer-state and update RMS, resource use, and training curves locally. `evidence.json` summarizes all six fits after every cohort is ready. If training runs before rollout collection finishes, it writes `archive-evidence.json`; a second fit invocation then reuses its checkpoints to score the fresh rollouts.

The paper policy checks one final TRM grid on each case still open after the menu. The stopping policy checks each cycle and commits the earliest verified grid. Per-policy verifier checks and shared-block calls are reconstructed from the full diagnostic replay, which scores every cycle for every case. Timings apply to the diagnostic run. Verifier-call counts exclude operator internals and include the raw proposal check whenever the parser emitted a candidate.

The Windows launcher reserves 1.5 GiB of available physical memory. Its Job Object caps host commit at 256 MiB during preparation, 768 MiB for the CPU GGUF rollout, and 1,280 MiB for the matched fits. The GGUF runner uses one CPU thread and zero GPU layers. The launcher returns exit code 75 if admission times out or the reserve is crossed. Rerunning a phase resumes its saved rollout records or training checkpoints.
