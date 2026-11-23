; An exchange uses two chained verifier receipts for the same proposal.
; The first removal may leave old violations. The endpoint must be feasible and strictly better.

(= (VerifiedStopExchanges $start)
   (match &self
      (, (PublicRank $start $before)
         (CheckedExchangeEdit $id $start remove $drop Accepted $mid)
         (CheckedExchangeEdit $id $mid add $add Accepted $end)
         (ExchangeEndpoint $id $end Feasible)
         (PublicRank $end $after))
      (let True (BetterPublicRank $after $before) (ContinueWithExchange $id $drop $add))))
