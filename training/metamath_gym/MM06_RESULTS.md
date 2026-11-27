# MM06 results: goal-ranked DFS establishes the next baseline

The [MM06 corpus](https://github.com/patrickdugan/MathCan/actions/runs/45921857627/artifacts/15163819605) was extracted from source ordinals 2401–4200 of the pinned set.mm. Its complete 4,200-theorem portable prefix passed C and Python verification. The local scoped parser independently matched 1,346 root scopes, signatures, initial stacks, all accepted tuples and the exact census of 2,450,036 primitive transitions. The 1,377 valid tuples include 31 alternatives to the reference. No acceptance disagreement or false completed certificate was observed. The candidate tuples themselves have the upstream Python and scoped local checks, not a separate external C check of every tuple.

## Family test (207 roots)

| Transition cap | Hash DFS | Goal DFS | Beam 4 | Memo DFS |
|---:|---:|---:|---:|---:|
| 8 | 2 | 0 | 0 | 2 |
| 16 | 5 | 6 | 1 | 5 |
| 32 | 21 | 52 | 10 | 21 |
| 64 | 41 | 89 | 27 | 41 |
| 128 | 76 | 128 | 54 | 76 |
| 256 | 112 | 154 | 77 | 112 |
| 512 | 136 | 179 | 95 | 136 |

At cap 512, goal DFS completes 179/207 certificates at 34,761 primitive transitions. Hash DFS completes 136 at 56,460; all 136 also appear among goal DFS successes, with 43 goal-only successes. Memo DFS matches hash DFS exactly on test; it saves 48 transitions on train. Beam4 completes 95, with 51 unsolved roots that terminate after beam pruning and 61 budget-unresolved. Four-hole test: goal DFS 41/62, hash DFS 19/62, beam4 6/62. All costs include failed transitions and the fixed runs after the prefix; the prefix, exhaustive audit, ranking overhead and memo-key computation are outside the search budget.

## Coupling diagnostic

Zero roots have a set of accepted completion tuples whose coordinate projections form a strictly larger Cartesian product. Most roots have exactly one solution, which makes this test vacuous for them. This does not imply that transition legality is independent of prior choices. The corpus measures real branching difficulty but supplies no positive witness for non-Cartesian valid-choice coupling.

For MM07, register a distinct diagnostic corpus before any further model fitting: enforce two independently valid completions with a failed coordinate recombination; separately measure conditional legality across reachable prefixes, exact state collisions, and search rank regret. Keep an explicit synthetic positive-control fragment separate from the natural set.mm cohort. Compare against goal DFS at matched budgets before any neural or BPD claim. The MM06 family test has been spent and remains public.

## Provenance and reproducibility

The exact runner executed this turn is archived in mm06_branching.py.gz. SHA-256 of the decompressed file: 913c2c22bff5edca3a40eb44d374a4f82e872706efcc9c287afbb95e41d279f5. The earlier protocol records a different runner digest; those bytes were unavailable locally. The new file independently implements its four policy descriptions. The result is a protocol reimplementation, not a byte-identical reproduction of the earlier registered runner. The receipt summary records hashes of the full local receipt and per-root results; these larger files are retained in the conversation attachments, not committed here.

```sh
gzip -dc training/metamath_gym/mm06_branching.py.gz > /tmp/mm06.py
cp training/metamath_gym/mm04_registered.py.txt /tmp/mm04.py
python /tmp/mm06.py github-actions-artifact-15163819605.zip --mm04 /tmp/mm04.py --out /tmp/mm06
```

Lower-budget metrics are prefixes of the deterministic 512-transition runs. The total local run took about 6.6 seconds in the available container; this is not a matched wall-time benchmark. No BPD intervention or neural training has been performed. All previous MathCan/EvalMath/DataGoblin CI checks were green when this continuation started. New archive commit CI has not yet been observed.
