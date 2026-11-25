# Reproduce the Aleph tournament

Run from the DataGoblin repository root with DataGoblin on `PYTHONPATH`, CPU PyTorch 2.10, NumPy, psutil and PyYAML. Public exports additionally use jsonschema (Draft 2020-12) and ReportLab for SVGs. Use a writable pytest temporary directory outside the research checkout. The reported fits used Windows, Python 3.12 and one low-priority processor per worker, with accelerators hidden. Core numbers select indices in the process's allowed affinity set. On Linux the wrapper uses a virtual address-space cap; allow for the CPU runtime's mapped libraries. Default cell memory is 3 GiB.

**Inputs.** The native mixed corpus has manifest SHA-256 `a0d2dfbe5a93e7630c60e1c182dccabc03d60efea376eb65503b62e812c66383`. Fetch, unpack and validate it at `runs/placement/mixed` using the [pinned binary archive and intake commands](../aleph_rlt_placement/RESUME.md). For the prior frozen research population, recreate the [v1 receipt gym](../aleph_rlt_placement/RECEIPT_PROTOCOL.md), [v2 coordinated checkpoints](../aleph_rlt_placement/RECEIPT_FOLLOWUP_PROTOCOL.md) and [preceding readout panel](../aleph_rlt_placement/RECEIPT_READOUT_PROTOCOL.md), or supply the original local archives. The fair decoder excludes previous held frame hashes. Use the same parent manifest with all descendants. The output ZIP contains public results and checkpoint identities. Exact trained weight bytes and secret panel draws are retained with the operator.

**Local experiments.** Use a separate fresh output root for independent replication. Each example below is serial and uses the same single core. Fresh scoring consumes validated frozen checkpoints. In the continual experiment create the private seed only after training finishes. Keep it outside the repository. The archive includes commitments to each study's realized panels. A fresh independent draw can change held scores. The exact finite oracle and its Lean proof are deterministic.

```sh
export PYTHONPATH=src
D=training/aleph_next
O=../aleph-next
P=runs/placement/mixed
python training/aleph_rlt_placement/gym.py --smoke

python "$D/bounded.py" --core 0 --wall 7200 architecture_lab.py train --out "$O/arch"
python "$D/bounded.py" --core 0 --wall 7200 architecture_lab.py score --out "$O/arch"
python "$D/bounded.py" --core 0 --wall 3600 arch_spectral.py --out "$O/arch"
python "$D/arch_atlas.py" "$O/arch"

python "$D/bounded.py" --core 0 --wall 5400 decoder_lab.py train --v2 ../receipt-v2 --out "$O/decoder"
python "$D/bounded.py" --core 0 --wall 3600 decoder_lab.py score --v2 ../receipt-v2 --prior-readout ../receipt-readout/results.json --out "$O/decoder"
python "$D/bounded.py" --core 0 --wall 1800 decoder_pooled.py --v2 ../receipt-v2 --prior-readout ../receipt-readout/results.json --out "$O/decoder"
python "$D/decoder_release.py" --out "$O/decoder" --figure "$O/decoder/decoder-utility.svg"

python "$D/bounded.py" --core 0 --wall 1800 nonlinear_lab.py certify --out "$O/nonlinear"
python "$D/bounded.py" --core 0 --wall 3600 nonlinear_lab.py train --out "$O/nonlinear"
python "$D/bounded.py" --core 0 --wall 1800 nonlinear_lab.py score --out "$O/nonlinear"
python "$D/nonlinear_summary.py" --out "$O/nonlinear"

python "$D/bounded.py" --core 0 --wall 3600 continual_lab.py train --base ../receipt-v2 --out "$O/continual"
python -c "import pathlib,secrets; p=pathlib.Path('../aleph-next/continual/private.seed'); f=p.open('x'); print(secrets.randbits(256),file=f); f.close()"
python "$D/bounded.py" --core 0 --wall 1800 continual_lab.py score --base ../receipt-v2 --out "$O/continual" --seed-file "$O/continual/private.seed"
python "$D/continual_release.py" "$O/continual"
python "$D/continual_figure.py" "$O/continual/atlas-input.json" "$O/continual/retention.svg"

python "$D/bounded.py" --core 0 --wall 10800 scale_lab.py --out "$O/scale" --pack "$P"
python "$D/scale_release.py" --out "$O/scale" --pack "$P" --figure "$O/scale/frontier.svg"
python "$D/bounded.py" --core 0 --wall 10800 study.py --pack "$P" --out "$O/mixed192" --width 80 --steps 192 --r 2 --arms prelude_reread site_damp --seeds 137 163 191 --roots 48 --probe-roots 6
python "$D/mixed_release.py" --new "$O/mixed192" --controls runs/placement/study
python "$D/bounded.py" --core 0 --wall 600 research_stream.py --v2 ../receipt-v2 --decoder "$O/decoder" --out "$O/research-stream-v2"
python "$D/atlas.py" --runs "$O" --out "$O/atlas" --source-commit HEAD
python -m pytest -o addopts= -q --basetemp ../aleph-pytest-temp
ruff check .
```

The mixed replay expects archived direct and global damping. On a fresh machine, run the same mixed training command with `--out runs/placement/study --arms direct global_full_damp` before summarizing. The code in the pinned source commit defines the reference controls. Some legacy training contracts fingerprint raw file bytes. Preserve checkout line endings, CPU library versions and artifact bytes to resume the exact campaign. An independent clean run generates a new traceable source contract. Throughput and sampled RSS are machine-specific.

**Strict model.** Execute `python training/aleph_rlt_placement/fullscale.py --r 2 --candidate global_full_damp` to allocate meta tensors, verify both forward shapes and count physical weights. Use `--r 3 --candidate pass_rlt` to audit the alternate work ledger. The planning commands do not start an optimizer or a paid accelerator.

Verify the research stream independently with `python training/aleph_next/verify_research_stream.py ../aleph-next/research-stream-v2`. Run `packed_probe.py` via the bounded driver with the new manifest path and its recorded SHA-256. The exploratory coalition counterexample is deterministic and uses the standard library. [Joint-query verifier](joint_query.py)
