; Rank complete, constraint-respecting repairs by distinct goal coverage.
!(match &self
  (Rollout $id Closed Eligible WithinBudget UniqueLineage CurrentRevision DependenciesMet ConflictFree $covered $mentions $quality $negCost)
  (Offer $id $covered $quality $negCost))
