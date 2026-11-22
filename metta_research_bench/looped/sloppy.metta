; Local curation checks size, cost and eligibility, and counts evidence mentions.
!(match &self
  (Rollout $id Closed Eligible WithinBudget $lineage $revision $dependencies $conflict $covered $mentions $quality $negCost)
  (Offer $id $mentions $quality $negCost))
