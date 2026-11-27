# MM-GYM-04 results

Local execution completed against the registered archive. All 6,648 candidate acceptance decisions agree with the scoped upstream Python oracle. The independent local parser also matches all root scopes, candidate signatures, prefix stacks and immediate-legal sets. The parser trusts the externally verified prefix and skips proof bodies; it does not independently establish their theorems. No neural optimization or BPD intervention took place.

| Budget | Challenge correct / 200 | Repaired | Regressed | Suffix transitions | Completed certificates |
|---:|---:|---:|---:|---:|---:|
| 0 | 163 | 0 | 0 | 0 | 98 |
| 1 | 180 | 17 | 0 | 102 | 152 |
| 2 | 187 | 24 | 0 | 176 | 175 |
| 4 | 190 | 27 | 0 | 269 | 184 |
| 8 | 194 | 31 | 0 | 423 | 188 |
| 16 | 199 | 36 | 0 | 587 | 196 |
| 32 | 199 | 36 | 0 | 677 | 197 |

All arms charge the same 2,400 initial candidate applications on challenge. Suffix transitions count both successful and failed primitive steps, simulated from cached full traces; they are not independent external verification calls. At 16 steps/root, 199 choices are correct, 196 of them complete, and 3 of the 4 unresolved choices would ultimately be valid. Public challenge is spent, and no held-out generalization claim follows.

Reproduce with Python 3.10+: python training/metamath_gym/mm04_registered.py.txt /path/to/github-actions-artifact-15136757263.zip --out mm04.json

Exact archived source is intentionally stored as a text file to preserve the pre-run SHA-256; Python can execute it directly. The runner is a small exploratory research instrument. Next gate: a prospectively frozen family split, early-stop and serial-vs-round-robin scheduling controls, and only then a learned/BPD budget allocator measured against the best deterministic policy. Do not train on the public challenge misses.

Previous MM03 hosted tests: 635 passed, 13 skipped; full CI failed six lint diagnostics. A separate commit addressed those six diagnostics; rerun pending.
