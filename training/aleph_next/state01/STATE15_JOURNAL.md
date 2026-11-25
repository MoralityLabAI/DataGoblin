# STATE-15 · Lean kernel replay of the archived STATE-10–14 certificates

On 25 November 2026, pinned Lean 4.19.0 verified **1,364 original theorem goals** and accepted all 136 separately checked negative-control contexts. Appending the registered invalid transport to each context produced 136 type-mismatch rejections. The release covers the public synthetic Horn and Nat/List rewrite grammars. Training-corpus admission and both paid Aleph runs remain on hold. [Public verifier run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45605910185)

| Archive | Original declarations checked literally | Original goals checked through typed steps | Valid contexts | Invalid transports rejected |
| --- | ---: | ---: | ---: | ---: |
| STATE-10 | 168 | 0 | 72 | 72 |
| STATE-11 | 353 | 0 | 64 | 64 |
| STATE-12 | 192 | 0 | 0 | 0 |
| STATE-13 | 278 | 41 | 0 | 0 |
| STATE-14 | 284 | 48 | 0 | 0 |
| Total | **1,275** | **89** | **136** | **136** |

Multiple solver paths can establish the same statement. The counts measure coverage of the named archive entries.

The long-certificate compiler preserved the original theorem signature and every printed equality and proof right-hand side. It checked adjacent endpoints before emitting one typed lemma per rewrite and a balanced tree of `Eq.trans` applications. For 8,039 archived rewrite steps, Lean audited all 15,989 helper lemmas and all 89 root theorems individually. Among the 1,364 roots, 168 depend on no axioms and 1,196 depend only on `propext`. The verifier admitted only the registered `propext`, `Quot.sound` and `Classical.choice` axiom set. A missing, duplicated or unrecognized axiom audit fails its query.

## Measured public Linux costs

Eight standard public Ubuntu runners used the same release revision `6caaee842e94`, with one Lean task worker per invocation. The complete workflow took 215 elapsed seconds including archive restoration, setup, scheduling, artifact transfer and audit. Query wall times below add across simultaneous workers. Child CPU covers the whole Lean process, including parsing, elaboration, kernel checks, diagnostics and startup.

| Query class | CLI invocations | Summed wall seconds | Summed CPU seconds | Median wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Literal positive batches | 206 | 195.460 | 197.531 | 0.676 |
| Typed long certificates | 89 | 637.372 | 687.479 | 4.184 |
| Valid control contexts | 136 | 26.544 | 26.439 | 0.183 |
| Rejected transports | 136 | 29.928 | 29.974 | 0.211 |
| Total | **567** | **889.304** | **941.423** | |

The largest literal archived theorem occupied 509,363 UTF-8 bytes. Its 562 original steps took 104.709 seconds as typed lemmas on Linux. Charging one constant per successful proof step would miss process startup, term structure and the measured costs of rejected attempts. A new routing study can charge actual query and search costs on its target host. The historical weight of 32 work units per proof step has no measured conversion to these seconds.

## Local singleton calibration

Separate Windows Lean 4.19.0 processes checked three public positive anchors per stage at the 10th, 50th and 90th percentiles by source bytes, plus the first and last signed control pair per applicable stage. All 26 expected outcomes occurred. The 15 positive probes had median 1.398 seconds wall and 1.266 seconds CPU, with a 26.942-second wall maximum. The four rejected transports had median 2.001 seconds wall and 1.461 seconds CPU. The four valid contexts had median 1.445 seconds wall and 1.367 seconds CPU. Three empty Lean invocations had median 1.288 seconds wall and 1.141 seconds CPU. Every observation includes the local process cost under this shared host's cache and scheduling conditions. [Singleton receipts](STATE15_SINGLETON_COSTS.csv)

The initial nine accepted singleton probes took another 15.021 seconds of process wall time. A six-step balanced compiler calibration took 1.878 seconds of wall and 1.844 seconds of CPU time. They were development checks and are separate from the 567-query Linux cohort.

## Retained raw-term trials

The original serial driver submitted eight literal declarations per batch, used a 180-second Lean process cap and awarded credit only when the whole batch passed. Both raw trials reached long STATE-13 and STATE-14 proofs. Each retained four measured batch timeouts. A timeout leaves every declaration in that batch uncredited. The complete sharded suite checked the registered negative controls separately. [Raw-trial ledger](STATE15_RAW_ATTEMPTS.json)

| Serial trial | Completed query receipts | Declarations attempted / credited | Charged wall seconds | Charged CPU seconds |
| --- | ---: | ---: | ---: | ---: |
| Windows local | 136 | 1,080 / 1,048 | 2,310.543 | 1,731.094 |
| Public Linux | 137 | 1,088 / 1,056 | 1,444.362 | 1,446.171 |

The Windows driver had no overall clock limit. Following full verification by the eight shards, we let the active query finish, retained its receipt and stopped before launching STATE-14 offset 48. The span from creating the first query file to the last durable checkpoint was 2,317.895 seconds. The Linux driver reached its 25-minute job cap. GitHub recorded 1,517 elapsed job seconds including cancellation and cleanup. Its per-query charges cover the 137 completed receipts. The final in-flight query has no process-time receipt. [Original public raw trial](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45603824863)

## Archive recovery and verification

The earlier public bootstrap job failed during Python setup after 7 job seconds. Its successor spent 67 job seconds and failed the original STATE-11 SHA-256 pin before running the kernel. On Windows the historical ranker also recovered the original solved counts but exchanged two alternative proofs of goal 8. Its LF-normalized positive archive hashed to `7db30239ddb88007f015a00d19436ea3e29b9bad4fa132e879539ef46a06d914`. Recombining existing proof traces recovered the exact registered archive `9702ca64b5a04240fc6043c53ad1a71f41e164e745d725bf3e335972f48f78d6`. The compact witness reconstructs all of its bytes deterministically and also reproduces the original negative-control pin. [Bootstrap run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45597210958) · [Archive reconstruction failure](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45597300121)

Windows and Ubuntu independently regenerated the same five prepared theorem sources and the same packaged negative controls. The Linux prepared manifest SHA-256 is `74c8de419b21ad0e3d7e84607175a2e73ad903aadb52d42ad9316f1a9edda6b7`. A separate local pass read all 1,142 public query receipts, logs and shard manifests, recomputed source hashes, parsed every axiom audit, checked each control and reproduced the public summary byte for byte. Its SHA-256 is `4e7ac41c8a6e4bbde8fc59bad254acbf7b1d8c7d4b70ec330c08046b75582032`. [Kernel summary](STATE15_KERNEL_SUMMARY.json)

## Reproduction

Run from the repository root with Python, the historical pinned CPU dependencies and Lean 4.19.0 on `PATH`. The public workflow pins Python 3.13.5, NumPy 2.3.5 and PyTorch 2.10.0 CPU. Use fresh output directories to retain earlier receipts.

```bash
STAGE=training/aleph_next/state01
python -X utf8 "$STAGE/state15_replay.py" --regenerate --prepare-only --archive-root work/state15-archives --out work/state15-prepared
for i in 0 1 2 3 4 5 6 7
do
  python "$STAGE/state15_shards.py" run --prepared work/state15-prepared --out work/state15-checks --index "$i" --count 8 --timeout 600
done
python "$STAGE/state15_shards.py" collect --prepared work/state15-prepared --out work/state15-checks --count 8 --summary work/state15-audit.json
python "$STAGE/state15_profile.py" --prepared work/state15-prepared --out work/state15-profile
```

To reproduce the literal batch trial separately, use the same archived inputs and a fresh output directory.

```bash
python training/aleph_next/state01/state15_replay.py --archive-root work/state15-archives --out work/state15-raw --batch-size 8
python training/aleph_next/state01/state15_audit.py --archive-root work/state15-archives --out work/state15-raw --summary work/state15-raw-audit.json
```
