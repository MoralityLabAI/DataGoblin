# MM-GYM-03 local development results

The verified corpus is frozen to MathCan [run 45858715061](https://github.com/patrickdugan/MathCan/actions/runs/45858715061), artifact 15136757263. All corpus artifact hashes were checked locally. A separate local implementation of the registered protocol (CPU PyTorch 2.10.0+cpu, one thread) reproduced the symbolic baselines and ran all nine neural arms. It is not a byte-identical execution of the committed pilot.py; exact source replay and hosted controls remain separate gates.

## Baselines

Local applicability agreed with the Python reference on all 6,648 candidates. Counts are top-1/3/5 hits; denominators 186, 199, 200.

| Method | Train | Dev | Public challenge |
|---|---:|---:|---:|
| First bank candidate | 24/65/101 | 18/58/84 | 18/55/84 |
| Train reference frequency | 55/125/156 | 25/40/77 | 28/54/91 |
| Prefix legality | 128/166/182 | 141/169/179 | 122/172/182 |
| Symbolic goal matching | 166/178/185 | 165/186/192 | 163/190/195 |

Symbolic and neural arms each use 1,902/2,346/2,400 local checks per train/dev/challenge split. Full replay counts are hypothetical stop-on-hit simulations using frozen oracle outcomes, not live calls. The symbolic challenge consumes 200/256/272 simulated suffix checks at top-1/3/5. The random expected top-1 challenge accuracy is 0.08625.

## Neural arms

All arms have 2,849 parameters and 960 updates. Fixed depths 1/4/8, seeds 109/113/127; no early stopping. Feature inputs include locally instantiated outputs and training-label frequency, as registered. This is a recurrent bank ranker, not a transformer or a native DataGoblin learner.

| Seed | Depth | Train hits @1/3/5 | Dev hits @1/3/5 | Challenge hits @1/3/5 | Challenge simulated calls @1/3/5 | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| 109 | 1 | 176/186/186 | 160/176/182 | 161/184/190 | 200/264/292 | 0.8905 |
| 109 | 4 | 184/186/186 | 157/172/182 | 157/181/186 | 200/269/304 | 1.5627 |
| 109 | 8 | 183/186/186 | 160/175/183 | 156/188/192 | 200/268/290 | 2.1659 |
| 113 | 1 | 177/186/186 | 161/176/181 | 159/183/190 | 200/266/295 | 0.8375 |
| 113 | 4 | 183/186/186 | 164/177/184 | 163/186/190 | 200/259/286 | 1.366 |
| 113 | 8 | 181/186/186 | 167/178/183 | 160/185/190 | 200/261/288 | 2.0587 |
| 127 | 1 | 174/186/186 | 162/175/181 | 162/182/190 | 200/263/293 | 0.8049 |
| 127 | 4 | 185/186/186 | 159/174/182 | 159/180/187 | 200/267/301 | 1.3654 |
| 127 | 8 | 178/186/186 | 166/177/183 | 159/184/189 | 200/263/292 | 2.0863 |

The symbolic policy wins over every depth on mean challenge top-1 and top-5. One depth-4 run ties symbolic top-1. No depth provides a positive mean incremental benefit. Do not promote the neural ranker. Development top-1 improves slightly at depth 8; the challenge does not reproduce that direction.

Next: freeze a dependency-family grouped corpus before training, and test repair after the symbolic policy's errors with a suffix-aware candidate state. Preserve a symbolic fallback; compare matched prefix and suffix-verifier budgets, candidate counts and model updates. Public source-order results cannot establish unseen-family generalization. No BPD intervention, native TRM, or cross-formalization was run.

Evidence: independent local implementation of fixed protocol; full set.mm and rewritten proof prefix verified externally by MathCan; alternative labels only Python-scoped checks. No theorem discovery claim.
