# ALG-DESCENT-11: witness-withheld Horn proof planning (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/8af9d2066df289daf720ff2ab541ac94497f4fe0). CPU development on synthetic public acyclic Horn rules, no solution witness or exact teacher values supplied at evaluation. Full graph is public. This is finite proof planning, not cryptographic discovery.

## Reproduction
From repo root, with PyTorch:
    SRC=src/datagoblin/research/alg_descent_v11
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/proof_planning.py"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/audit.py"

Python 3.13.5, PyTorch 2.10.0+cpu, one torch thread. Source git blobs match locally executed files: proof_planning.py 7741baa9fd0fce589d9228b7a085ec3ccfdbc05c; audit.py 691acfdfc2644368f408ec24e40cfae08015f023. Replay archive supplied separately with checkpoints, receipt, public rule graphs, audit and original logs. SHA256 proof_planning.py d2decf4c1576d7488b259f0364d300940ef206962186d66db685037f07a38a2f; audit.py 020623c338306f6c3f26185052b4026ee84b1338cb88eee7c77dafaeb746a7be; receipt.json 1bac91fbcf85e82a1a6d6780680f05b3b68aace98b7acc5745b840f60bb681ab; models.pt fc394c5fda4231ab9a3c26325ab87268870ae8f461f1c59a91ea719cd249ede0; graphs.json 04fb8cd865972681f1b89c32930515d6a5fa4e7eb925165a55bdd5dc0a64eac6; audit.json 13557d75bf08ddc77f5b9cf7ca1f15aea0a8782b4f21a9c706657547e5b5ab9c. Timing-containing receipts have run-specific hashes.

48 train roots, 3,242 state examples from train-only exact Bellman oracle. 16 IID and 24 OOD roots, eight OOD roots per family. Train and IID alternating family cycles correlate with graph size (uniform 10/12, clustered 11/13); OOD family/size rotated by block. First run failed during evaluation because an empty successor batch fell through to a missing symbolic arm key; fixed with early return and reran unchanged seeds, arms, budgets and gates. MLP 433 parameters, tied 361; three seeds 19/41/67, both 30 epochs, AdamW .003, smooth L1 on normalized residual over FF. Recurrence is internal to each state estimate, not persistent search history. Model estimate is clipped below by admissible hmax but is not necessarily admissible. Full graph, known facts and public heuristic features available to all relevant arms; no privileged solver at evaluation. Offline oracle used for reference only.

## OOD results
| Arm | Closed | Proof/opt (solved only) | Expansions (all) | Main rule probes (all) | Mean search ms |
|---|---:|---:|---:|---:|---:|
| UCS | 2/24 | 1.0000 | 61.208 | 2095.21 | .362 |
| hmax A* | 24/24 | 1.0000 | 20.333 | 691.79 | 3.433 |
| hadd | 24/24 | 1.0323 | 5.833 | 199.38 | 1.409 |
| FF | 24/24 | 1.0137 | 5.875 | 200.75 | 1.381 |
| weighted FF | 24/24 | 1.0439 | 5.708 | 194.96 | 1.313 |
| MLP | 72/72 | 1.009337 | 8.292 | 283.31 | 2.270 |
| tied | 72/72 | 1.009337 | 7.667 | 261.00 | 2.307 |

Both learned arms PASS registered point-estimate gate against hmax A* (strongest symbolic arm by closure then cost): .9337% mean proof inflation, 59.2%/62.3% expansion reductions, each 21/24 root wins and 3 ties. Tied minus hmax expansions mean -12.6667, descriptive paired root-bootstrap 95% [-17.8472,-8.1111]. MLP mean -12.0417, interval [-17,-7.625]. Both proof-ratio differences interval [0,.0260271]; statistical 1% noninferiority not established. Root averages pool seeds, not independent seed/root samples. Tied vs MLP expansion mean -.625, 11/11/2 wins/ties/losses, interval [-1.2083,-.1111], equal seed-averaged proof costs on all roots; parameter counts and architectures differ, no causal recurrence claim. Tied vs FF +1.7917 expansions, 1/9/14, proof ratio -.004398 (2/22/0). FF faster. Shared/uniform OOD tied and MLP exactly optimal on every run; clustered mean ratio 1.028011. IID all 176 searches close; hmax, MLP and tied all exactly optimal; expansions 8.438/6.396/6.479.

Timing single randomized-order search per root/arm excludes graph/oracle prep and independent final proof replay. hmax timing includes full shared public heuristic features (hmax/hadd/FF and summaries), not optimized standalone A*. Main rule probes count transition scans, not heuristic work. No universal speedup. Search budget 64 expansions, positive costs, exact proved-set deduplication. 22 failures all OOD UCS, 418/440 closures; proof ratio excludes failed rows.

## Audit and stronger baseline
Set-based successor audit checked 48,831 Bellman states (5,074 train; 43,757 eval). Independent set-based certificate replay all 440 receipts, 2,251 proof steps and three negative fixtures (bad rule ID, duplicate head, missing premises). Rule registry digest covers rule table and is checked within trusted root context; scope should bind initial facts and goal as well for a portable receipt. Independent model inference/search ranking replay not done.

Post-hoc direct FF extraction on spent roots closed all 24 OOD goals, 12 optimal, mean proof/opt 1.149131, mean 0.01714 ms construction excluding graph grouping/final replay. This is a constructive incumbent, not a new registered competitor. Next test anytime exact search with direct FF incumbent, admissible frontier lower bound and learned priority; charge heuristic construction and replay. Require equal exact guarantees, compare optimized symbolic baseline and match memory/expansion budgets. No native BPD/MMS, Lean verification, general proof-search speedup, or DLP claim.