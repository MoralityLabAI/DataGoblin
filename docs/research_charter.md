# Research charter

The DataGoblin constitution is an executable integrity policy, not a proof of alignment. It exists to make research decisions inspectable, bounded, and difficult to self-certify.

Hard rules require disjoint train/evaluation namespaces, independent evaluation, solver or invariant certification, label provenance, finite resources, denied-by-default network and subprocess access, preregistered metrics, complete seed reporting, held-out capability measurements, falsifiers, controls or ablations, content hashes, and fail-closed behavior. Unsafe operational mechanics are outside v0 scope.

Soft objectives rank otherwise valid proposals by held-out gain, transfer, information gain, sample efficiency, mechanism coverage, diversity, calibration, reusability, simplicity, and compute efficiency. Soft objectives never override a hard violation.

Negative, null, aborted, and failed results remain first-class ledger records. They may narrow hypotheses and inform later experiments without being coerced into positive findings.

