---
name: metta-curate-sources
description: Audit MeTTa source provenance and rights, assign source-root splits, and prepare source-preserving SFT or RLT curation proposals for independent review.
---

# Curate MeTTa sources

1. Pin the repository, commit, blob, path, excerpt boundaries, and content
   hash. Record license and consent evidence. Keep upstream sources read-only
   and flag missing rights or provenance for review.
2. Group derivatives, reformattings, alternate views, and repairs by source
   root before assigning splits. Reserve development and frozen roots and
   their derivatives from training. Keep frozen feedback out of selection.
3. Preserve each presentation ID in both SFT and RLT exports. Count those
   two views as one presentation. For the DataGoblin trajectory experiment,
   enforce the four-presentation cap per source root across the exports.
4. Verify the claims separately. Native replay checks the behavior of the
   query. DataGoblin's independent custody and admission checks determine
   whether the candidate can be released for training. Carry unresolved
   claims forward as pending.

From DataGoblin, generate synthetic policy examples and check the existing
corpus contract with:

```sh
python -m MeTTa.curation_gym --out runs/metta/curation
python -m pytest -q tests/unit/test_metta_corpus.py
```

Return a proposed disposition, source identities, split and presentation
counts, and the evidence or receipt supporting each decision. Keep private
source content and frozen labels out of the public response.
