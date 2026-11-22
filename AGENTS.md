# DataGoblin contributor guide

- Preserve the distinction between public typed research state and private backend workspace.
- Treat proposer evidence as a claim; only independent or environment-owned verification may hard-promote it.
- Keep lifecycle transitions monotone and evaluated records immutable.
- Run `python -m pytest` and `ruff check .` before handoff.
- Keep demos offline and deterministic. Never commit secrets, model weights, hidden evaluation seeds, or generated runs.
- Keep HRM-Text optional: never auto-download weights, persist full activations by default, or run a real checkpoint
  outside an operator-provided hard resource-cap wrapper.
- Upstream repositories are read-only references. Do not copy unlicensed source or mutate upstream files.
- When the human provides research taste for storyworld or videogame experimentation, use `Agonist/SKILL.md` and its offline `Agonist/gym.py` verifier before proposing a live Inkling worker.
- For student-uplift curation experiments after AGN-01 world verification, follow `Agonist/skills/agonist02/SKILL.md` and preserve evaluation-root custody.
