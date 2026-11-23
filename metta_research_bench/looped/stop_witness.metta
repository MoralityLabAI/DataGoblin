; Read public ranks and environment check receipts. Only a feasible improvement can defer STOP.

(= (BetterPublicRank (Rank $a $b $c $d) (Rank $w $x $y $z))
   (if (> $a $w) True
       (if (== $a $w)
           (if (> $b $x) True
               (if (== $b $x)
                   (if (< $c $y) True (if (== $c $y) (< $d $z) False))
                   False))
           False)))

(= (StopDeferralWitnesses $state)
   (match &self
      (, (PublicRank $state $before)
         (CheckedStopNeighbor $state $id Accepted Feasible $after)
         (PublicRank $after $rank))
      (let True (BetterPublicRank $rank $before) (ContinueWith $id))))
