; Original MeTTa curation teaching fixture: fact/query substrate.
; Facts describe a fictional intake, not permissions to ingest data.
; The DataGoblin EtiquettePolicy remains the final decision authority.
;
; A reviewer can match source, license, lineage and intake state using &self.

(Candidate sample_a)
(Candidate sample_b)
(Candidate sample_c)
(Candidate sample_d)
(Candidate sample_e)

(Source sample_a hyperon_reference)
(Source sample_b metta_examples)
(Source sample_c metta_storyworld)
(Source sample_d hyperon_reference)
(Source sample_e metta_examples)

(License sample_a MIT)
(License sample_b MIT)
(License sample_c MIT)
(License sample_d UNKNOWN)
(License sample_e MIT)

(Custody sample_a DISCOVERY_ONLY)
(Custody sample_b DISCOVERY_ONLY)
(Custody sample_c EVALUATION_ONLY)
(Custody sample_d QUARANTINE)
(Custody sample_e QUARANTINE)

(HasProvenance sample_a True)
(HasProvenance sample_b True)
(HasProvenance sample_c True)
(HasProvenance sample_d False)
(HasProvenance sample_e True)

(NativeVerified sample_a False)
(NativeVerified sample_b False)
(NativeVerified sample_c False)
(NativeVerified sample_d False)
(NativeVerified sample_e False)

(Split sample_a train_candidate)
(Split sample_b train_candidate)
(Split sample_c ood_reserved)
(Split sample_d train_candidate)
(Split sample_e dev_reserved)

(SameLineage sample_a sample_b)
(SameLineage sample_b sample_a)

; Retrieve proposals by joining facts. No admission or pruning is executed.
!(match &self (License $id MIT) $id)
!(match &self (, (Candidate $id) (Custody $id DISCOVERY_ONLY)) $id)
!(match &self (, (Split $id ood_reserved) (NativeVerified $id False)) $id)
!(match &self (, (HasProvenance $id False) (Custody $id QUARANTINE)) $id)
!(match &self (SameLineage $a $b) ($a $b))
