# Machine-review fallback: Batch01 closed

27 September 2026.

## Why the process changed

The first risk-based review batch originally ended with one physics-convention disagreement awaiting a human subject expert. A human adjudicator is not available to this project. We therefore changed the release rule instead of pretending that model agreement is expert certification:

- any disagreement that could change the target is quarantined;
- a quarantined family is replaced by a separately reviewed family from the same subject and exam cell;
- failed and conflicting reviews remain part of the private audit record;
- the result is labelled **machine reviewed**, never human-expert certified.

## Result

| Outcome | Families |
| --- | ---: |
| Original families accepted after blind review and deterministic checks | 11 |
| Original families quarantined | 1 |
| Same-cell replacement families accepted | 1 |
| Active families after replacement | 12 |
| Active linked exercises after replacement | 36 |

The held item was a resonance-tube problem with a convention dispute. Its replacement was a different JEE Advanced physics problem whose numeric result was independently derived and checked against the source's accepted answer interval. The old family is absent from the active candidate and remains preserved in the private quarantine record.

## Full-candidate verification

- 300 independent families and 900 linked exercises remain active.
- The aggregate structural audit reports zero failures.
- The rebuilt component passes exact image transport, assistant-only loss-mask, decoded-target and 2,048-token preparation checks with the pinned Qwen3.5-4B processor.
- Rebuilding the component is byte-stable.
- The refreshed candidate-index SHA256 is `ba20e64c8007ed7b8661f3ffbcd71d2fbdc332d5cf484f4fdea8ed66c287037b`.
- Optimizer updates: **0**.

## Reviewer limitations discovered

The attempted third-review path was itself informative. One free reviewer produced unusable output on both image and text inputs. Another did not support image inspection, but produced usable structured decisions for three answer-free text transcriptions and hit its output cap on two. Those two cases were admitted only because both original image reviewers agreed and the remaining caveats were resolved by complete deterministic derivations. This exception does not apply when an unresolved caveat could change the target.

## What this establishes

Batch01 no longer blocks indefinitely on unavailable human adjudication. It establishes a conservative process for removing unresolved examples while preserving subject/exam balance. It does **not** establish that all 300 families are reviewed, that the dataset is ready for training, or that any model improved.

The next gate is another immutable high-risk batch from the refreshed risk register. A training release remains blocked until the planned review coverage and release checks are complete.

Private question text, images, answers, derivations and reviewer transcripts are excluded from this repository.
