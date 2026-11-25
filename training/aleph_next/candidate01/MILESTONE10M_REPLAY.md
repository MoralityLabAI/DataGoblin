# Replay the direct-A 10M corpus

Run from the DataGoblin checkout. The native reader streams IDs and masks without model weights. Use Python 3.12 and the original CPU dependencies, including tokenizers 0.22.2 for data generation. The bounded launcher accepts an index into the worker's allowed CPUs, a memory cap in MiB and a wall limit in seconds. The commands here run serially on one allowed CPU. They use Bash syntax. Keep the training store separate from the confidential evaluation store.

## Read the shipped corpus

For the self-contained operator check, run `python -I -B verify_bundle.py --receipt reproduced.json` at the root of a fresh extraction. For the repository reader, extract the operator ZIP to `../aleph-10m-extracted` (or use the archive's own README). The five train lanes live under `mix/`. Run from the repository root with the SHA-256 below. A replay must use a new output receipt file. No model or tokenizer download is needed for the read.

~~~sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
W=training/aleph_next/bounded.py
M=candidate01/milestone_mix.py
P=../aleph-10m-extracted
SHA=be25522ded491e0c0c3f81d87d786034b46733e1ccbe2b9cea2dafc47b48abf4
python "$W" --core 0 --memory-mib 1024 --wall 600 "$M" replay --base "$P/mix" --sha256 "$SHA" --receipt "$P/independent-replay.json"
~~~

The ordered-packs digest for each of 39 update groups binds lane name, actual length, all token IDs, segment IDs, position IDs and shifted labels. Read each lane's JSONL separately to check the packed IDs and labels directly against its source rows. All prompts, document boundaries and padding take their original masks. Closing and reopening the native readers at update 19 must reproduce the remaining 20 update groups and complete terminal cursor. Local wall and CPU time are observations and may vary. [Native packer and segment semantics](../../aleph_rlt_placement/packing.py) · [Independent replay](milestone_mix.py)

## Rebuild and verify the five additions

These commands require the previous operator store at `../aleph-candidate01`, the frozen mixed pilot, the original CONFIRM14 panel and source cache, and the pinned LibreTexts shard plus tokenizer. Prior train-only archives alone are insufficient for custody reconstruction. The historical data and license acquisition paths are documented in [the starter replay](REPLAY.md). Use a fresh destination for a build. Archive the original output if testing a revised generator.

~~~sh
O=../aleph-10m-rebuilt
OLD=../aleph-candidate01
mkdir -p "$O/extra"
CPU() { python "$W" --core 0 --memory-mib 1024 --wall 900 "$@"; }
CPU candidate01/milestone_research.py self-test
CPU candidate01/milestone_research.py build --base "$OLD" --out "$O/extra/custom_research"
CPU candidate01/milestone_research.py verify --base "$OLD" --out "$O/extra/custom_research"
CPU candidate01/milestone_checked.py --base "$OLD" --out "$O/extra"
CPU candidate01/milestone_checked.py --base "$OLD" --out "$O/extra" --verify
python "$W" --core 0 --memory-mib 3072 --wall 900 candidate01/milestone_text.py build --base "$OLD" --out "$O/extra/general"
python "$W" --core 0 --memory-mib 3072 --wall 900 candidate01/milestone_text.py verify --base "$OLD" --out "$O/extra/general"
python "$W" --core 0 --memory-mib 3072 --wall 900 candidate01/milestone_text.py audit --base "$OLD" --out "$O/extra/general" --audit-output "$O/general-source-audit.json"
CPU "$M" freeze --prior "$OLD" --extra "$O/extra" --out "$O/mix"
NEWSHA=$(python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$O/mix/index.json")
CPU "$M" replay --base "$O/mix" --sha256 "$NEWSHA" --receipt "$O/replay.json"
~~~

The frozen index also commits to the source verification receipts. A clean rebuild that times local verification again can produce a different index digest even when every training row is identical. Compare source shard hashes, per-update token/mask digests, quotas and target counts. An exact operator-bundle replay must also match the original index and terminal cursor.

For training, use the single merged index as entry point. The native additions, attributions and verifier transcripts in the bundle provide provenance for the same rows. Keep those sidecars out of the sampler. The deterministic within-lane ordering is frozen for this first leg. A future continuation requires the same consumed prefix, including the pending offsets at 10M.
