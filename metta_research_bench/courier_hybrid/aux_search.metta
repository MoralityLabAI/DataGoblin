; Enumerate successful suffixes under a bounded primitive fuel list.
(= (HybridFind $state $fuel)
   (let (HybridDone $terminal) (HybridAcceptStop $state) (HybridRoute ())))

(= (HybridFind $state $fuel)
   (if (== $fuel ())
       (empty)
       (let* ((($tick $rest) (decons-atom $fuel))
              ((HybridTransition $action $next) (HybridSuccessors $state))
              ((HybridRoute $tail) (HybridFind $next $rest)))
             (HybridRoute (cons-atom $action $tail)))))
