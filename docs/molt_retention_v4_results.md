# MOLT-04 development results — 2026-11-26

Status: descriptive development evidence only. Source head `e9caa548f7889e33359ba7bcb029dc66f8dfd738`.
[Dedicated CI run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45839286318) completed successfully, including focused pytest, Ruff, training, evaluation, and receipt artifact upload.

The fixed study used 1,024 TRAIN roots, 128 short and 128 long development worlds, four training seeds (31, 37, 41, 43), and four fitted recipes per seed. It produced 86,016 scored rows. Raw rows digest: `782b0de8d809fd5cea03436733b1a4a6846eee4fa13ad6f7feedf24d68b48251`. The Actions artifact holds the raw development report subject to its retention period; no generated run files are committed here.

## Selected development slice

Capacity 2, one verifier call; entries are verified success rates. Learned arms average the same 128 worlds over four training seeds per column. These columns are paired within world and crossed with model seeds.

| Policy | Short clean | Short revised | Long clean | Long revised |
|---|---:|---:|---:|---:|
| FIFO | 0.359375 | 0.3359375 | 0.2578125 | 0.2421875 |
| Random | 0.3125 | 0.2578125 | 0.2578125 | 0.2421875 |
| Scope | 0.53125 | 0.5 | 0.2734375 | 0.2421875 |
| Single | 0.62890625 | 0.556640625 | 0.384765625 | 0.3671875 |
| Single reinvest | 0.638671875 | 0.5703125 | 0.380859375 | 0.37109375 |
| Tied3 | 0.63671875 | 0.572265625 | 0.41015625 | 0.392578125 |
| Tied3 shuffled | 0.2734375 | 0.255859375 | 0.16015625 | 0.154296875 |
| Generator rank | 0.6484375 | 0.6015625 | 0.4453125 | 0.4296875 |
| Hindsight | 1 | 1 | 1 | 1 |

Tied3 minus single-reinvest: -0.1953125 and +0.1953125 percentage points on short clean/revised; +2.9296875 and +2.1484375 points on long clean/revised. The slice was printed by the frozen driver; all capacities, budgets, and variants remain in the report. These observations do not establish a population-level advantage or a mechanistic causal result. The generator reference is optimal in expectation under the fixture law, not a pointwise bound on a finite sample. Hindsight has privileged information.

## Next discriminator

MOLT-05 should preregister a paired contrast on fresh development roots, then test a frozen BPD-inspired intervention on the scorer's actual recipient-state write. Full reconstruction parity, task competence, matched displacement and Jacobian-based controls, and independent calibration precede any steering claim. Separate model and task-root variation; use EvalMath preflight for a prospective confirmation design. Do not treat this panel as sealed evidence.

The current resource comparison matches forward cell-example visits only. Actual FLOPs and peak process-tree RSS are missing; serialized receipt payload is not RAM. The fitted scorer is a tiny tied NumPy head and the host computes the final answer from retrieved receipts. Real checkpoint evaluation and native TRM/MoLT/BPD integration remain separate.
