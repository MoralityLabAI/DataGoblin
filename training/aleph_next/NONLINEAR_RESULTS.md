# Evidence value and nonlinear uplinks

At four transmitted bits, D reached mean Brier .13059 on fresh mixtures of the programs it learned, compared with .13264 for the optimal linear-code/Bayes-reader control. D beat that control on two of three seeds. C and the MLP did so on zero seeds. The registered gate requires all three. On previously unused programs, the recurrent systems' valid uploads made their own matched receivers worse. Keep the nonlinear solver as an external control and retain the negative architecture result.

Each isolated site has eight possible worlds. The pretraining search certified a unique nonlinear one-bit winner for each of 16 public objectives and 32 additional transfer objectives. Objectives on which a coordinate, a GF(2) row, any real affine threshold, or a direct query answer was optimal were rejected. Two objectives on the same observations require different messages. On the public programs, the smallest separation from the combined shortcut class was .00116 Brier and the largest was .01095. The proof and exact census describe the channel and computational assumptions. [Derivation](NONLINEAR_MATH.md) · [Lean certificate](NonlinearReceipt.lean) · [Registration](NONLINEAR_PROTOCOL.md)

## Eight decentralized researchers

The table gives verified Brier per late decision, averaged equally over the two declared objectives. The native LDT selector prices endpoint gains and sees the deployment objective before choosing workers. It sees no realized query or private observation. The nonlinear optimum uses the exact public query distribution. Uniform allocation activates a random public subset. C, D and MLP each use a matched sender and receiver. The mixed population cycles the three trained architectures across sites and uses a different architecture as each site's reader. Learned rows average three matched seeds.

| Transmitted bits | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Ideal side only | .18993 | .18993 | .18993 | .18993 |
| Best linear code and allocation | .17157 | .15560 | .13264 | .11232 |
| Optimal nonlinear code and allocation | .16432 | .14368 | .11326 | .08782 |
| Native LDT allocation with exact local codes | .16432 | .14373 | .11326 | .08782 |
| Uniform allocation with exact local codes | .17717 | .16441 | .13888 | .08782 |
| Code chosen without the deployment objective, native allocation | .16527 | .14559 | .11693 | .09298 |
| C | .17757 | .16284 | .14001 | .12270 |
| D | .17277 | .15592 | .13059 | .11180 |
| MLP | .18882 | .17741 | .16794 | .15936 |
| Heterogeneous learned population | .18388 | .16914 | .15300 | .13648 |
| Exact verified utility per bit | .02561 | .02313 | .01917 | .01276 |

The four-bit D seed scores were .12898, .12840 and .13440. C and D reproduced all 16 optimal sender codes per seed on these public programs. Their conditional readers leave a gap to the .11326 optimum. Direct value ranking matched or slightly beat native LDT allocation. The LDT comparison therefore supports objective-aware allocation over blind allocation while leaving direct ranking as the stronger cheap control. At eight slots every allocation policy buys every worker. The local centralized two-bit control also saturates earlier than a raw three-bit upload would suggest. Two independent pair parities plus any one verified coordinate recover all three measurements, and both linear and unrestricted two-bit encoders reached zero loss. Extra four- and eight-bit capacity gives no further local benefit.

## New mathematical programs

A private draw after hashing all nine models selected one of two additional precertified sets. The interventions and conditional proof obligations changed within the same two mathematical grammars. Four-bit Brier was .24872 for C, .25463 for D and .24534 for the MLP. Ideal side-only was .19397, the strongest linear system reached .13585, and the nonlinear optimum reached .11681. Relative to each model's own matched no-message reader, realized utility per bit was −.00522 for C, −.00775 for D and +.00047 for the MLP. Each architecture averaged 0.5 optimal senders of eight across objectives and seeds. The sender was trained to imitate public code tables using categorical site identity. This outcome leaves transfer to newly defined local predicates unresolved and demonstrates negative utility from the recurrent uploads on this panel.

All fits used 640 matched Adam updates of 64 draws. C and D have 6,880 trainable parameters and two shared visits. The MLP has 1,601 parameters. The nine fits consumed 266.0 summed wall seconds on one CPU core. The results reflect a finite enriched generator, two objectives per site and one fresh program-set draw. They do not estimate the prevalence of nonlinear opportunities among actual research claims.

## Reproduction

From the DataGoblin repository, set PYTHONPATH to include src and use a Python environment with NumPy and CPU PyTorch. Supply an unused output directory and an allowed CPU core index. Keep the private evaluation seed and individual model weights in the local run directory.

```powershell
$d = 'training/aleph_next'
$out = '../aleph-next/nonlinear'
python "$d/bounded.py" --wall 300 --core 1 nonlinear_lab.py certify --out $out
python "$d/bounded.py" --wall 1200 --core 1 nonlinear_lab.py train --out $out
python "$d/bounded.py" --wall 300 --core 1 nonlinear_lab.py score --out $out
python "$d/nonlinear_summary.py" --out $out
python -m pytest -q tests/test_aleph_next_nonlinear.py
```

The local manifest records model, stream, oracle and source hashes. The original public summary SHA-256 is `61f090743656fb914ed4bfdcd783b6a07403ff297e0e67bf9ac3b7ed2647f8e3`. The training certificate SHA-256 is `aed078f7607ff780c036e6c0a477600ed73e15c665df5315a21a243cc3a2ae1d`. An independently drawn seed changes numerical mixture scores. The atlas and gym records contain aggregate measurements and provenance.
