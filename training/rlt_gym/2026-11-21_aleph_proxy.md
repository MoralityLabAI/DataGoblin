# Aleph proxy for verified repair and stopping

The proxy is a 192-weight ridge controller over the frozen RLT reservoir's candidate pool. It ranks existing proposals with public input features and checked failure history. Fitting uses 1,239 passing outcomes, 146 failed outcomes, 146 failed-to-valid repair pairs, and 37 matched-context preferences from the Aleph pack. The fit covers 1,230 train roots with at most four selected views per root. The environment independently rechecked 2,967 training receipts before selection. [Source campaign](2026-11-21_campaign.md)

The trial selected check confidence and repair margin on 84 calibration roots. The selected settings were `-0.15` and `0.10`. Eight calibration configurations used 1,016 checker calls in total. The final comparison used 336 fresh roots, evenly divided between ordinary and structural OOD and balanced across all seven families. The freeze excluded every exact lineage found in the campaign, its teacher suites, and the preliminary smoke suite. The held-out cohort remained frozen while the depth-4 comparator was configured with its own planned clock.

## Held-out results

| Policy | Verified | Ordinary / OOD | Verifier calls | Recurrent visits | Checks per solve |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed depth 4 | 128/336 | 57 / 71 | 336 | 1,344 | 2.63 |
| Fixed depth 8 | 121/336 | 53 / 68 | 336 | 2,688 | 2.78 |
| First verified native proposal | 147/336 | 69 / 78 | 356 | 1,710 | 2.42 |
| Repair ordered by native score | 336/336 | 168 / 168 | 639 | 639 | 1.90 |
| Learned repair order | 336/336 | 168 / 168 | 455 | 455 | 1.35 |

The fixed-depth policies check once at the endpoint. First-verified stopping skips checks of an already checked candidate on the same task. Both repair policies can try unchecked pool alternatives under the same four-check limit. The learned controller used **184 fewer checks and recurrent visits** than the heuristic at equal coverage, a 28.8% reduction. It used 99 more checks than first-verified native stopping and verified 189 additional roots.

Every learned visit issued a check. The verifier triggered every accepted stop. The learned gain came from candidate ordering, including 134 reranks before an observed failure and 75 accepted repairs after a failed check. All pools in this cohort contained a reachable valid candidate. Both repair policies reached that coverage ceiling.

The learned policy computed 1,805 additional lightweight candidate scores. Summed measured rollout time was effectively tied with the repair heuristic at 13.58 s and 13.68 s, respectively. The 2,122 held-out checker calls were replayed against the saved candidates. Those audit rechecks are separate from policy work.

## Reproduction and resource receipt

```powershell
python scripts/run_aleph_proxy_windows.py --campaign CAMPAIGN --out NEW_RUN --ramcap-package LCGRT_ROOT
```

Pass `--cohort FRESH_TASKS.json` to reuse the exact frozen cohort. The run directory stores the fit, calibration sweep, per-root checks, summary and resources. The completed single-thread CPU run took 149.61 s and peaked at 119 MiB of job commit under the 768 MiB hard cap. Its summary SHA-256 is `ac192f7740b2286be28761d26dae6f697ed80d9d1b026ad4201da82792b962f4`.
