# ZKL-14A: lazy polynomial column allocation

SatsHedgeZero source pin: 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb. Identical graph ordering, budget of 24
offered columns, fresh seed block 13301–13308, four test families, F11 and
BN254 Fr. Each arm receives a separately constructed identical problem.
Lazy allocation changes only the time at which polynomial coefficients
are expanded; every proposed identity is still checked independently.

Local reproduction (search engine exact Git blob; local verifier uses the
same relevant functions, not the complete remote poly_certificate.py blob):

| Metric | Eager | Lazy |
|---|---:|---:|
| Certified tasks | 24/32 | 24/32 |
| Materialized columns | 13640 | 393 |
| Materialized coefficient terms | 26840 | 723 |
| Offered columns | 393 | 393 |
| Coefficient work proxy | 2494 | 2494 |
| Score evaluations | 13640 | 13640 |

All paired certificates, reduction counts and inversions match. Essential
controls remain uncertified. The scoped hosted workflow reruns using the
full pinned checker. Key enumeration and ranking still examine the entire
candidate index pool; the column and coefficient counts are allocation
proxies, not elapsed time or RSS. No kernel replay or production authorization.
