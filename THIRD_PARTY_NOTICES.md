# Third-party notices

DataGoblin was designed after read-only inspection of the following research repositories:

- MoralityLabAI Control-Harness, commit `c938da0f3d4320201d4539ccf3c639891ff5b180`.
- MoralityLabAI HybridTRMLDT, commit `978c4929b872edbc3c1e3a5f0201ff89fd1c87fb`.

No clear root license grant was found in either inspected checkout, so DataGoblin contains clean-room implementations of compatible concepts and interfaces rather than copied source. The repositories influenced typed provenance, public/private state separation, conservative ingestion, deterministic receipts, and claim-boundary conventions.

Runtime dependencies retain their own licenses: NumPy and PyYAML. Development tools include pytest, Ruff, mypy, and pytest-cov.

The optional HRM-Text adapter targets the released `sapientinc/HRM-Text-1B` checkpoint and the corresponding
`sapientinc/HRM-Text` source repository. Both were published under Apache-2.0 at the time of inspection. DataGoblin does
not redistribute their code or weights. The adapter uses a clean structural hook boundary and treats the meaning of
recurrent states as a research hypothesis.
