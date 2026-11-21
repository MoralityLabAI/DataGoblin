# Six-pass Inkling holdout on the RTX 5080

The reduced-memory configuration completed all 14 cases in the first
frozen holdout batch. All 14 answers reached independent verification,
and six passed. This batch used a fixed depth of six and covered all
seven task families.

| Task family | Passed | Attempted |
| --- | ---: | ---: |
| Arithmetic programs | 0 | 2 |
| Constraint assignment | 0 | 2 |
| Evidence retrieval | 2 | 2 |
| Experiment planning | 2 | 2 |
| Finite-state planning | 0 | 2 |
| Graph routing | 1 | 2 |
| Table transformations | 1 | 2 |
| **Total** | **6** | **14** |

The ordinary test split passed 4/7 and the out-of-distribution split
passed 2/7. The cohort's one- and three-pass controls are pending.

## Runtime and resource measurements

The [streamed placement](streaming_weights.md) kept physical blocks
0–5 and 20–23 on CPU, with all 12 recurrent core blocks on the 5080.
The loader streamed **9,052,198,144 bytes** and measured **8,633 MiB**
of Vulkan model buffers. The guarded job finished in **731.6 seconds**
with peak job commit of **11.39 GiB**. Live minima were **6.57 GiB**
available system RAM, **7.79 GiB** available commit, and **6,878 MiB**
free VRAM. The existing resource guards held for the full batch.

## Trace and frozen SAE readouts

The capture has **38,136 finite vectors** from **5,448 prompt tokens**.
Every token has a complete prelude plus six indexed core boundaries.
The earlier frozen SAE encoded the final prefill token on each boundary.
Its last-token reconstruction MSE averaged 3.78 at pass two and stayed
between 5.36 and 5.42 from passes three through six. Mean hidden update
norm across prompt tokens declined from 39.69 at pass four to 23.60 at
pass six. These are state measurements on the held-out batch.

The local `native-holdout-core-offload-audit-02` capture contains answer
verdicts and the run provenance. The separate
`native-holdout-core-offload-sae-audit-02` capture holds readouts,
per-root statistics, and hashes of the native trace and frozen SAE.
The backend intervention stages use their own matched controls.
