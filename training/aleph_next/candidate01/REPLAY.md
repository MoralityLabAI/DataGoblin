# Reproduce the final candidate decision

Run from the PR checkout. Commands use Bash syntax. On this operator machine the equivalent Windows paths are `../aleph-state15/lean-4.19.0-windows/bin/lean.exe` and `../aleph-next/lean-4.28.0-windows/bin/lean.exe`. Use a fresh output directory for reconstruction. Retain any previous bundle and its logs.

## Check the existing operator release

Use Python 3.12 with CPU Torch 2.10.0, NumPy 2.3.5, tokenizers 0.22.2, PyYAML and psutil. On Windows the original admitted and sealed data live under `../aleph-candidate01`. The following commands are read-only apart from ordinary test temporary files. The final command intentionally returns 2 while supply and target-backend gates remain open.

~~~sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
D=training/aleph_next/candidate01
S=training/aleph_next/state01
O=../aleph-candidate01
python training/aleph_rlt_placement/gym.py --smoke
python -m pytest -q tests/test_candidate01_production.py tests/test_candidate01_curation.py tests/test_aleph_allocation_final.py tests/unit/test_aleph_state15_replay.py
python "$D/reserve.py" verify "$O/private-evaluations"
python "$D/snapshot.py" check "$O" --supplement --sha256 cb2495da7078bdb5320fb348a007744cb2473711cedca6d34341d6622a04cb1d
python "$S/allocation_evidence.py" verify --evidence ../../outputs/ALLOCATION_EVIDENCE.zip
python "$D/readiness.py" --base "$O" --measure-capacity --require-ready
~~~

## Reconstruct the kernel and routing evidence

Lean 4.19.0 at revision `6caaee842e94` verifies the historical replay and final solver trajectories. The public Linux jobs describe the distribution download, exact toolchain and disjoint matrix. Use the extracted official Lean distributions, or assign L19 and L28 directly to the pinned local executables. Use an available core on an idle machine. The historical Linux source preparation used Python 3.13.5, NumPy 2.3.5 and CPU Torch 2.10.0.

~~~sh
TOOLS=../candidate01-toolchains
mkdir -p "$TOOLS"
for v in 4.19.0 4.28.0; do
  curl -fL --retry 3 "https://github.com/leanprover/lean4/releases/download/v$v/lean-$v-linux.tar.zst" -o "$TOOLS/lean-$v.tar.zst"
  tar --zstd -xf "$TOOLS/lean-$v.tar.zst" -C "$TOOLS"
done
L19="$TOOLS/lean-4.19.0-linux/bin/lean"
L28="$TOOLS/lean-4.28.0-linux/bin/lean"
K=../candidate01-replay-kernel
python "$S/state15_replay.py" --regenerate --prepare-only --archive-root "$K/archives" --out "$K/prepared"
for i in 0 1 2 3 4 5 6 7; do
  python "$S/state15_shards.py" run --prepared "$K/prepared" --out "$K/joined" --count 8 --index "$i" --lean "$L19" --timeout 600 || exit 1
done
python "$S/state15_shards.py" collect --prepared "$K/prepared" --out "$K/joined" --count 8 --summary "$K/audit.json"
~~~

For a fresh routing run, each command below uses one worker, one allowed CPU and the resource wrapper. The original host used disjoint workers and saved source and log checksums for the common cost ledger. New wall measurements can change deadline outcomes and fit targets. The shipped evidence verifier replays the original decisions from the original measured results. The synthetic tournament generator is public and retired for future private model selection.

~~~sh
T=../candidate01-replay-allocation
worker() { python "$S/allocation_worker.py" --core 0 --wall 7200 "$@"; }
worker allocation_replay.py --out "$T" --lean "$L19" --split train
worker allocation_routes.py freeze --out "$T"
worker allocation_replay.py --out "$T" --lean "$L19" --split held
worker allocation_routes.py score --out "$T"
python "$S/allocation_evidence.py" pack --root "$T" --evidence "$T/evidence.zip"
python "$S/allocation_evidence.py" verify --evidence "$T/evidence.zip"
~~~

## Reconstruct the corpus and restart fixture

Restore the historical mixed pack using the commit and unpack commands in [RESUME.md](../../aleph_rlt_placement/RESUME.md). The prior privately drawn CONFIRM14 text panel must match its registered manifest, `82761739ea4f078bd1631dba2851be6551109692fbfe5166295429094fe45c51`. The frozen metadata index binds exact upstream bytes. The external dataset and tokenizer are public downloads. Fetch them once and keep them in the operator store.

~~~sh
N=../candidate01-replay-data
P=runs/placement/mixed
mkdir -p "$N"
curl -fL --retry 3 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json' -o "$N/tokenizer.json"
curl -fL --retry 3 'https://huggingface.co/datasets/common-pile/libretexts_filtered/resolve/204d01e40af72b3b6d3f926e1f6148ee9ed3970a/libretexts-0000.json.gz' -o "$N/libretexts.json.gz"
python "$D/text_intake.py" --source "$N/libretexts.json.gz" --tokenizer "$N/tokenizer.json" --native "$P" --history ../aleph-confirm14/text --out "$N/prose"
python "$D/curation.py" build --pack "$P" --lean "$L28" --tokenizer "$N/tokenizer.json" --count 1536 --out "$N/verified"
python "$D/curation.py" verify --pack "$P" --lean "$L28" --out "$N/verified" --recompile
python "$D/curation_custody.py" --verified "$N/verified" --retrospective "$K/prepared" --lean "$L28" --out "$N/custody-verified"
T_ORIGINAL=../aleph-allocation/final
python "$D/tournament_admission.py" --source "$T_ORIGINAL" --base "$N/verified" --tokenizer "$N/tokenizer.json" --lean "$L28" --out "$N/tournament-train"
python "$D/reserve.py" freeze "$N/private-evaluations"
python "$D/snapshot.py" freeze "$N" --supplement
python "$D/restart.py" --manifest "$N/prose/train/manifest.json" --sha256 f5ecac1c41d1e651e9174131812fe410d0d094de36fef4efc7871658c9879e38 --out "$N/cpu-restart"
~~~

The supplement accepts timing evidence only from the measured Windows Lean 4.19.0 binary. Use the original train-only evidence at T_ORIGINAL for that admission step, including the raw singleton logs. Fresh Linux router timings are an independent benchmark. A fresh private reserve has a different digest by design. Source paths, Lean builds, compiler logs, OS timing and storage serialization may also change commitments. To check the original release, use the original operator store and [SUPPLY.json](SUPPLY.json). To check a new snapshot, hash its `snapshot-final/index.json` and pass that digest to `snapshot.py check --supplement`. Keep hidden keys, final questions and targets in the custody store. Access to the train-only ZIP is sufficient to consume the five training lanes. Full custody verification also requires the evaluator's private store.

The backend and supply gates for a real launch are listed in [READINESS.md](READINESS.md).
