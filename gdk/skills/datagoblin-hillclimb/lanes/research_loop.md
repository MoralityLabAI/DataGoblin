# Lane: research_loop (other DataGoblin suite runs)

This lane has no metric: `decide` closes each move as `explored`. Use it to learn what the rest of the
suite does before proposing moves elsewhere.

- `dg_demo`: the research loop end to end (Scout hypotheses, Alchemist recipe, Rat critique,
  ResearchMembrane decision, builder tickets, independent verifier, NumPy learner, ledger, report).
- `dg_validate`: validate a DataRecipe (`recipe`, default `examples/recipe.good.yaml`) against the
  constitution: split custody, reference solver, independent verifier, builder permissions. A
  nonzero exit with a rule name is the expected result for a bad recipe.
- `gym_demo`: the architecture training gym's mock controls through the frozen evaluation firewall.
- `etiquette_demo`: the collection-etiquette pipeline's custody lanes.

Report what a run showed that changes your next move in another lane.
