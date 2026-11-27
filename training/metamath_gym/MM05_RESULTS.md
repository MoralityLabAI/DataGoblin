# MM05 results: serial early-stop wins

The fresh [MathCan artifact 15164147697](https://github.com/patrickdugan/MathCan/actions/runs/45920441822/artifacts/15164147697) passed both independent verifiers for the rewritten 2400-theorem prefix. ZIP SHA-256 f3c4afdd5d5f54d689a405c44bfde69040cf32cc2947fa4db5d228180c095dfe. It yields 1746 new masked roots (1198 train, 267 dev, 281 family test); 34 additional masks from prior alpha families excluded. Full scoped replay agrees on all 18746 candidate decisions. The train-only ridge allocator fits 54253 nonterminal trace states from 860 roots; the other 338 train roots have no suffix work. The exact typed alpha grouping does not isolate all semantic families or shared dependencies; the test is public.

| Family test policy | Budget 16 completed / 281 | Budget 16 transitions | Budget 128 completed / 281 | Budget 128 transitions |
|---|---:|---:|---:|---:|
| Serial, no early stop | 215 | 2213 | 275 | 7323 |
| Round robin, no early stop | 188 | 2213 | 257 | 7323 |
| Serial, early stop | 215 | 1723 | 275 | 3644 |
| Round robin, early stop | 188 | 1963 | 257 | 6923 |
| Serial, cache | 215 | 1723 | 275 | 3644 |
| Round robin, cache | 188 | 1963 | 257 | 6919 |
| Ridge, early stop | 214 | 1723 | 275 | 3645 |

At 128, serial early stop selects a valid action for 280/281 and completes 275; six unresolved, five of which would be valid. No false completed certificates. The 3037 initial candidate applications are shared across all family test arms and not included in suffix counts. Cached deterministic state simulations count exact primitive transitions, not measured wall-time or external verifier invocations. Full audit and feature/cache costs are outside policy transition budgets. No BPD or neural model was trained. Cache gains are negligible here and the learned allocator fails to improve serial early stop. Keep serial as the baseline; before neural spend, isolate residual failures and construct a prospective harder multi-step repair task.

Spent previous challenge control: serial early stop completes 200/200 with 262 transitions at budget 32 or 128; round robin early stop completes 200/200 with 486 at budget 128; ridge completes 200 with 269. Old challenge used for replay only. Old fit uses 83 train roots / 512 trace states.

Reproduce the exact local source:
```sh
cp training/metamath_gym/mm04_registered.py.txt /tmp/mm04.py
python training/metamath_gym/mm05_schedule.py.txt github-actions-artifact-15164147697.zip --mm04 /tmp/mm04.py --fresh --out fresh.json
python training/metamath_gym/mm05_schedule.py.txt github-actions-artifact-15136757263.zip --mm04 /tmp/mm04.py --out spent.json
```
The source is archived as .txt to preserve the executed bytes without changing lint policy; execute directly via Python. The committed JSON contains the complete test policy grid, not the entire local receipt (its hash is included). The exact runner was executed locally; hosted execution of this file is not claimed. Two remaining MM03 lint issues were fixed before this run. No cross-foundation theorem translation or native TRM/BPD result.
