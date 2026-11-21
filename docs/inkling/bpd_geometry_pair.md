# Paired BPD geometry intervention on Inkling Q4

The native comparison tested aligned and orthogonal activation directions
at the final prefill token of pass two. Both directions targeted SAE
feature 121 at L2 dose 5.084054, with the same frozen prompts,
checkpoint, Vulkan runtime, placement, and three-pass decode settings.
Each of the seven roots ran untreated, sham, matched-random, and
selected-component arms in both geometries.

All **56 answers** reached the independent checker. The untreated and
sham replies agreed across geometries on all seven roots.

| Geometry | Untreated | Sham | Matched random | Selected component |
| --- | ---: | ---: | ---: | ---: |
| Aligned | 4/7 | 4/7 | 4/7 | 4/7 |
| Orthogonal | 4/7 | 4/7 | 3/7 | 3/7 |

The selected component and matched-random arms had equal solve counts
within each geometry. The aligned-minus-orthogonal solve contrast was
zero. For each geometry the scorer subtracted the matched-random distance
from the component distance to the untreated pass-three state. The mean
aligned-minus-orthogonal difference in this excess distance was **−1.80**
L2 units across the seven paired roots. These measurements describe the
response to the registered interventions.

Aligned and orthogonal execution took 416.9 and 392.0 seconds,
respectively, including model load. Peak job commit was 10.52 and
10.50 GiB under the existing guards. The local
`native-bpd-geometry-core-offload-02-paired-01` directory contains
the independently verified results, matched rows, and capture digests.
