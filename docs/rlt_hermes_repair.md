# Rectified feedback in the Hermes Campsite repair TRM

The October revision measured the effect of having a small TRM write a repaired Campsite grid. The original Hermes menu tries the raw proposal, a C-only projection, and a dual projection. It passed 63 of 148 archived greedy Qwen2.5-3B proposals. Appending a five-seed MeTTa-atom TRM ensemble raised the total to 93 of 148, with zero invalid commits. The public verifier owned every commit. [October paper revision](https://github.com/MoralityLabAI/morality-lab-site/blob/9477211c95cda2a689e6b04a8596cd09f8a51104/papers/MeTTa_Scaffolded_Repair_Curricula_for_TRM_Infused_Hermes_Skills__1_.pdf)

This pilot uses the original Campsite prompt, parser, projections, verifier, MeTTa cell features, and grid decoder. It scores every archived proposal and generates 36 new greedy proposals on a size-stratified subset of the paper's test puzzles. Those puzzles are held out from repair training. New generations on known puzzles measure transfer across proposers. A test on newly generated puzzles would be needed to measure transfer across puzzle distributions. [Qwen2.5-0.5B-Instruct Q4_K_M checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/6dd44a1fb35d11b5d1b28902876ce3cc9e882d0e/qwen2.5-0.5b-instruct-q4_k_m.gguf)

The three arms share the paper's 64-wide repair model, three answer cycles, and three latent updates per cycle. Direct feedback uses the shared block's answer endpoint each cycle. RLT interpolates from the current answer state toward that endpoint with weights `[1/3, 1/2, 1]`. Matched damping uses `[5/12, 5/12, 1]`, with the same total interpolation weight before the final cycle. All arms use the original latent-state update. The fit checks exact final-logit equality between the direct adapter and the paper's implementation.

Each arm sees the same 512 archived training proposals from 256 puzzles and receives 320 AdamW updates per seed. Every training target is a nearest legal solution verified by the original verifier. The training puzzles come from the paper's training pools, with one greedy proposal each from Qwen2.5-3B and Bonsai-8B. The pilot uses two independent seeds per arm and scores single models. The paper's ensemble score is a historical reference.

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
