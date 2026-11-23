; The environment supplies distances and checked edits for the current repair state.
; A shortest action must be accepted and advance by exactly one edit.

(= (VerifiedShortestActions $state)
   (match &self
      (, (RepairDistance $state $before)
         (CheckedRepairEdit $state $action $record Accepted $after)
         (RepairDistance $after $remaining))
      (let True (== $before (+ $remaining 1)) (RepairChoice $action $record))))

(= (VerifiedShortestActions $state)
   (match &self
      (, (RepairDistance $state 0) (CheckedRepairStop $state Accepted))
      (RepairChoice stop none)))
