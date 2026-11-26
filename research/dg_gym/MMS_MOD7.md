# MMS follow-on: existing affine and bijective MOD7 gyms

This uses the existing Core and sample generator; the original frozen pilots
and their results are unchanged. Explicit GRUCell equations expose the
recurrent affine. Output and gradient parity are tested against the original
fused cell. The zero-memory control for this follow-on also uses the explicit
cell. SVD maps are proxies; an exported learned BPD map must refer to the
exact recurrent affine and lineage. Physical cotangents are cached and used
through the current read/write map. No additional forward/backward pass.

```
PYTHONPATH=src:. python -m research.dg_gym.mod7_mms --gym bijective --seed 101 --updates 1200 --threads 2 --mms-mib 4 --mms-mode bpd --cpu-seconds 30 --out runs/mod7_mms_bpd
```

Repeat at 0 MiB and positive-budget observe/dense/sham, both with and without
concept supervision. Report fixed exposure and fixed CPU separately. The
soft CPU cap may terminate mid-curriculum; retain actual length exposure.
The prior bijective competence gate remains relevant, and no promotion or
mechanistic claim is made on this engineering fixture. Exact full-program
keys may have low repeat coverage on length-four training; report misses.
Tiny gradient rows and finite early-depth support make this a MiB-scale
diagnostic; do not allocate GiB merely to increase the reported RAM. Use
RMP's full-sequence tensor rows for the large-memory study. The budget covers
bank tensor payload and tags, not total RAM. Use the resource watchdog and
count external process-tree RSS, preparation and evaluation.

Map admission amendment: the absolute 0.5 weight cosine gate applies only to
a frozen exported map. Rank-truncated SVD explained energy is rank-dependent;
for example rank-16/128 random initialization can have cosine below 0.5.
SVD and native live maps require a nonzero reconstruction cosine, and all
feedback still requires positive current/historical gradient alignment.
Neither gate establishes causal fidelity.
