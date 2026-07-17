# DataGoblin contributor guide

- Preserve the distinction between public typed research state and private backend workspace.
- Treat proposer evidence as a claim; only independent or environment-owned verification may hard-promote it.
- Keep lifecycle transitions monotone and evaluated records immutable.
- Run `python -m pytest` and `ruff check .` before handoff.
- Keep demos offline and deterministic. Never commit secrets, model weights, hidden evaluation seeds, or generated runs.
- Upstream repositories are read-only references. Do not copy unlicensed source or mutate upstream files.

