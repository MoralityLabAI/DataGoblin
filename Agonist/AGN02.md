# AGN-02: research-selection and downstream student gym

AGN-02 advances the [AGN-01 research agonist](README.md) from verifying a proposed
counterfactual world to testing whether **a curator can choose training worlds
that make another learner more capable**. Its first implementation is an offline
public development experiment. It is a contract and scientific pilot, not a
trained 1B DataGoblin result or evidence of a working Inkling provider.

## Reproduce

From the repository root with Python 3.10+, numpy and the test extras installed:

    python -m Agonist.agn02 run --out runs/agonist02/dev50 --seed 17
    python -m Agonist.agn02 replay --archive runs/agonist02/dev50
    python -m pytest tests/unit/test_agonist02.py
    ruff check Agonist tests/unit/test_agonist02.py

The run uses fifty synthetic, independently identified **taste families**.
A caller may pass a source-attributed JSONL taste file with exactly 34 train
and 16 dev family roots using the AGN-01 schema:

    python -m Agonist.agn02 run --tastes /path/to/tastes.jsonl \
      --out runs/agonist02/from_taste

In this development fixture the source ID affects the generated mechanics
through a stable hash. The example source ideas are original and fictional.
Meaningful interpretation of an arbitrary new human idea requires the
separately authorized local Inky world-building worker. The present run
issues **zero teacher calls** and assigns **no measured teacher tokens**.

The cohorts are disjoint by taste family:

| Cohort | Roots | Purpose |
| --- | ---: | --- |
| Anchor | 10 train | Common supervised seed set for every learner |
| Candidate pool | 24 train | Curators choose 12 additional roots |
| Calibration | 8 dev | Estimate candidate utility for the learned surrogate |
| Evaluation | 8 dev | Evaluate trained students; isolated from curation and fitting |

All eight evaluation roots remain public synthetic development samples.
There is no sealed confirmation or cross-generator structural holdout.

## Three decisions from one evidence revision

For each canonical root, an exact environment produces two counterfactual
fictional worlds with one changed prerequisite and the same surface cue.
The agent is asked to:

1. **Act:** decide deploy or hold in the original public state.
2. **Revise:** decide again after an independently supplied prerequisite
   change. The correct action must reverse.
3. **Check or stop:** evaluate the expected utility of buying another
   source-freshness check before committing a later decision. The source
   change hazard, check price and mistake cost are public. A check can help
   only when resource and safety prerequisites permit deployment.

The environment computes the label for the third action from its public
contract: if structural prerequisites permit deployment, checking has
expected advantage in hundredths of a unit of
\`hazard_percent * mistake_cost - 100 * check_cost\`. Otherwise checking
costs \`100 * check_cost\`. Check iff the advantage is strictly positive.

The two world labels come from the AGN-01 exact action verifier. The
student receives one public decision state at a time. Labels are held
by the grader and used for supervised training only on anchor and
selected candidate roots. The surface cue is a heuristic confound.

## Matched curator arms

- **Anchor only:** common supervised anchor roots, for measuring what
  adding any pool examples does.
- **Random:** seeded random selection of twelve pool roots.
- **Fixed:** frozen ranking that favors wrong surface cues, positive
  verification actions and near-threshold resource margins.
- **Learned:** a small ridge utility selector trained on ten leave-one-out
  anchor utility measurements on the calibration cohort. Its training
  targets are the downstream student's accuracy changes. It is a
  research proxy for DataGoblin's future agonist selection head.

All non-anchor arms share the same candidate pool, exact oracle and 12-root
selection quota. Each student trains on 66 labeled decisions using 240 SGD
updates per seed. Three student seeds are matched by arm; the anchor-only
reference has 30 labeled decisions and also receives 240 updates. Every
selector has access to the same source corpus. The learned arm incurs
an additional eleven calibration-student fits and a ridge solve, which
must be accounted for when comparing total research cost.

Full episode completion requires all three decisions correct. Record
per-decision accuracy, per-root/seed whole-episode success, paired
learned-minus-baseline differences and a descriptive root bootstrap.
The eight evaluation roots are too few for a powered efficacy claim.

## Artifact contract

\`runs/agonist02/dev50/\` is a content-hashed experiment archive:

| File | Contents |
| --- | --- |
| roots.jsonl | Public synthetic worlds, causal revisions, checks and source custody |
| oracle_receipts.jsonl | Independent initial/revised/check actions and exact value of checking |
| selector_training.jsonl | Ten leave-one-out calibration utility targets |
| selections.jsonl | Exact chosen train roots and matched exposure counts |
| student_scores.jsonl | Per-root, per-seed, per-arm outcomes |
| manifest.json | Provenance, work counts, predictive control comparison and artifact SHA-256 |

Replay verifies digests and **independently recomputes** all labels,
candidate choices and trained student scores. It detects a relabeled
receipt even when the archive manifest hash has been rewritten.
Existing output directories are protected against silent overwrites.

## Interpretation and next research gate

The AGN-02 selector is a small trained **surrogate** with an intentionally
weak generalization probe. It cannot establish that DataGoblin 1B learns
research taste, that Inky improves world diversity, or that the student
benefit generalizes to unseen generator programs.

To advance toward those claims, use the AGN-01 teacher queue and
approved local Inkling service to generate executable worlds with
real token receipts. Add a versioned environment adapter for richer
storyworlds and games; retain exact or independent verifiers for their
actions. Compare learned DG curation against fixed and random policies
with identical teacher budget and candidate pool, then train students
at matched sample and optimizer budgets on entirely held-out
generator families. A preregistered EvalMath analysis should include
root-clustered intervals, failure rates, contamination audits and the
cost of fitting the selector.

All candidate training views remain \`training_ready=false\`.
No private seeds, real model activations, model weights, secrets,
provider responses or generated run archives are committed.
