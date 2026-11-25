# Replay the direct versus global constant damping confirmation

Use the DataGoblin research checkout at the commit recorded in the report, Python 3.12, CPU PyTorch 2.10, NumPy and the DataGoblin dependencies. Run from the repository root. Commands below use Bash syntax with one allowed CPU at a time. On the study host the six independent cells used separate logical CPUs. Only advance after the registered direct horizon fails. A serial run will take longer. Keep the generated checkpoints, roots, score capsules and private draw keys outside Git.

Use the immutable mixed corpus in training/aleph_rlt_placement/RESUME.md, with mixed dataset manifest SHA-256 a0d2dfbe5a93e7630c60e1c182dccabc03d60efea376eb65503b62e812c66383. The text commitment itself is 82761739ea4f078bd1631dba2851be6551109692fbfe5166295429094fe45c51. See training/aleph_next/REPLAY.md for the archived ALEPH-06 prefixes. Numerical hashes identify the particular published tensors used here. The historical mixed package must be unpacked into the expected directory before starting.

~~~sh
set -eu
export PYTHONPATH=src
D=training/aleph_next
P=runs/placement/mixed
PREFIX=../aleph-next/scale
O=../aleph-confirm14
mkdir -p "$O"
cpu() { python "$D/bounded.py" --core 0 --memory-mib 2048 --wall 7200 "$@"; }
python training/aleph_rlt_placement/gym.py --smoke
python "$D/confirm_lineage.py" --prefix "$PREFIX" --out "$O/prefix-lineage.json"
cpu confirm_speed.py --prefix "$PREFIX" --pack "$P" --out "$O/speed.json"

# Source files, source cache and original private text directory audited first.
python "$D/confirm_text.py" audit --pack "$P" --cache "$O/source-cache" --panel "$O/text"

for h in 64 128 256 384 512; do
  for a in direct global_full_damp; do
    for s in 137 163 191; do
      cpu anchor14_train.py --prefix-dir "$PREFIX" --pack "$P" --out "$O" --arm "$a" --seed "$s" --until "$h"
    done
  done
  cpu anchor14_train.py --out "$O" --select
  if test -e "$O/selection.json"; then break; fi
done
test -e "$O/selection.json"
for s in 137 163 191; do
  cpu anchor14_train.py --out "$O" --pack "$P" --arm global_full_damp --seed "$s" --score-locked "$O/selection.json"
done
python "$D/confirm_lock.py" --run "$O" --prefix "$PREFIX" --out "$O/fits.json"
cpu confirm_eval.py freeze --fit-manifest "$O/fits.json" --text-dir "$O/text" --pack "$P" --out "$O/eval"
cpu confirm_eval.py prepare --out "$O/eval"
for a in direct global_full_damp; do
  for s in 137 163 191; do
    cpu confirm_eval.py score --arm "$a" --seed "$s" --out "$O/eval"
  done
done
cpu confirm_eval.py reduce --out "$O/eval" --speed "$O/speed.json"
python "$D/confirm_text_check.py" --release "$O/eval" --out "$O/independent-text.json"
~~~

To reproduce original scores, retain the original frozen text directory, original private reasoning draw and source cache. For an independent replication use fresh output directories, build the authored panel with the fetch/freeze commands in CONFIRM_TEXT.md and commit to its new manifest before training. The original independent checker pins the original text manifest, so a separately registered replication should pin its own manifest there. The model comparison remains paired because the same new evaluation sample is shared by all six endpoints. The study uses per-byte nats and verified exact success. Report fresh family outcomes separately from the archived graph replay.

The receipt archive includes numeric development histories and the registered statistical decision, plus a second implementation's exact text check. Load eval/decision.json for all transparent cue scores, primary and sensitivity bounds, arm-level competence, per-seed results and family-level outcomes. The archive excludes raw private examples and numerical weights.
