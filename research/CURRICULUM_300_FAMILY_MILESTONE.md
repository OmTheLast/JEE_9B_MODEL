# 300-family 4B curriculum candidate milestone

26 September 2026.

## Status

The planned curriculum-scale candidate pool is complete: **300 independent train-reserved families** and **900 linked teaching views**. All 20 expansion batches were reviewed and closed separately, then checked together. **This is not yet a training release, expert-certified dataset, or model-improvement result.** Every private record remains `training_ready:false`, and no 4B optimizer update has run.

Private question text, images, answers, annotations, reviewer traces and prepared messages are deliberately excluded from this repository.

## Aggregate inventory

| Measure | Result |
|---|---:|
| Independent families | 300 |
| Linked teaching views | 900 |
| Atomic verification / method composition / solve-and-commit | 300 / 300 / 300 |
| Families per subject/exam cell | 50 |
| Atomic pass / fail | 150 / 150 |
| Source records reviewed for the 240-family expansion | 247 |
| Rejected source records / same-cell replacements | 7 / 7 |
| Qwen3.5-4B total tokens per view | 201–1,220 |
| Qwen3.5-4B assistant tokens per view | 47–646 |
| Views over 2,048 tokens | 0 |
| Optimizer updates | 0 |

The six cells are Mathematics, Physics and Chemistry crossed with JEE Main and Advanced. Three views from one question remain one sampling family; 900 views do not mean 900 independent problems.

## What the individual and collective reviews caught

Seven source records were excluded because their official disposition or preserved references did not support one defensible target: two marks-to-all items and five contradictory or multi-answer cases. They were replaced by independently reviewed, unused records from the same subject/exam stratum.

The review and replay process also caught:

- OCR errors in radical, superscript and option layout;
- correct calculations mapped to the wrong printed option;
- malformed or figure-blind external-review output;
- a numeric normalizer that changed an integer ending in zero;
- a generalized method-label default attached to unrelated image-derived solutions;
- seven accepted solve targets that still said a figure was missing while committing an option.

The affected batches were reopened and rebuilt. Superseded evidence remains in the private research workspace. The final aggregate auditor now blocks unresolved-figure language in solve targets.

## Collective technical checks

The final deterministic audit passed twice with identical outputs. It verifies:

- 300 unique parent IDs and family IDs, plus 900 unique view IDs;
- all parents remain train-reserved and have no recorded development/protected-test family overlap;
- all image paths and hashes;
- exact structured target and assistant JSON equality;
- no answer-key or reviewer-provenance language in model targets;
- no unresolved-image solve target;
- complete processor evidence for all 900 views, with image tensors, assistant-only loss masks and exact decoded targets.

These checks prove dataset structure and transport. They do not independently prove every scientific derivation or show that fine-tuning improves unseen JEE performance.

## Next gate

1. Independently review every high-risk family: manual image adjudications, option-label corrections, Advanced Chemistry, shared stems and diagrams.
2. Run a stratified expert audit of lower-risk families and expand review if its defect rate crosses the frozen threshold.
3. Freeze accepted corrections into a separately hashed training release with a family-aware sampler.
4. Prove one-update recovery and trained-weight export on the selected 4B route.
5. Run one conservative training exposure and compare unchanged versus adapted 4B under cell-level answer, procedure, preservation and output-mode stop gates.

The unchanged base remains the fallback. A lower training loss alone cannot select an adapter.
