; Extend the pinned native courier relation with two independent source bits.
(= (HybridSuccessors (HybridState $core $first $second))
   (let* (((CourierTransition $action $next) (CourierSuccessors $core))
          (True (or (not (== $action Deliver)) (and $first $second))))
         (HybridTransition $action (HybridState $next $first $second))))

(= (HybridSuccessors (HybridState (CourierState $site $permit $cargo) False $second))
   (let True (not (== $cargo Delivered))
        (match &self (HybridSource $site First)
               (HybridTransition AttestFirst (HybridState (CourierState $site $permit $cargo) True $second)))))

(= (HybridSuccessors (HybridState (CourierState $site $permit $cargo) $first False))
   (let True (not (== $cargo Delivered))
        (match &self (HybridSource $site Second)
               (HybridTransition AttestSecond (HybridState (CourierState $site $permit $cargo) $first True)))))

(= (HybridReplay $state $plan)
   (if (== $plan ())
       (HybridEnd $state)
       (let* ((($action $rest) (decons-atom $plan))
              ((HybridTransition $action $next) (HybridSuccessors $state)))
             (HybridProof $action $next (HybridReplay $next $rest)))))

(= (HybridAcceptStop (HybridState (CourierState $site $permit $cargo) $first $second))
   (let True (and (== $cargo Delivered) (and $first $second))
        (HybridDone (HybridState (CourierState $site $permit $cargo) $first $second))))
