# First 4B curriculum tranche

24 September 2026.

## Status

The first balanced training-data tranche for the approximately 4B program is prepared and has passed coordinator contract review plus processor-level integrity checks. **No 4B optimizer update has run.** The training safety lock remains active.

The underlying exam questions, images, answers, annotations, review pages and prepared tensors are private and are not included in this repository.

## Why the curriculum changed

Earlier 9B adapters mostly learned from long completed explanations. Training loss fell, but fresh-question performance did not reliably improve and some later checkpoints regressed. The new contract separates three behaviors that the diagnostics showed can fail independently:

1. **Atomic verification:** judge one proposed local step, explain the decisive issue and repair it when needed.
2. **Method composition:** choose the representation, conditions and bounded route without carrying out the final calculation.
3. **Solve, verify and commit:** execute only the necessary steps, check the decisive result once and emit the exact answer object.

The three views from one source question remain one independent family. They cannot be counted as three separate problems.

## Aggregate inventory

| Measure | Result |
|---|---:|
| Independent families | 60 |
| Linked teaching views | 180 |
| Families per subject/exam cell | 10 |
| Atomic pass / fail | 30 / 30 |
| Atomic target words | 15–62 |
| Method target words | 64–243 |
| Solve target words | 44–280 |
| Qwen3.5-4B total tokens per view | 214–1,036 |
| Qwen3.5-4B assistant tokens per view | 53–610 |
| Views over 2,048 tokens | 0 |
| Optimizer updates | 0 |

The six balanced cells are Mathematics, Physics and Chemistry crossed with JEE Main and Advanced. The atomic labels are also balanced within each cell.

## Defects caught before training

Seven retained local candidates documented the review process. The review caught and fixed:

- a planning prompt whose targets still contained completed calculations;
- an all-negative verification set that could teach a blanket rejection shortcut;
- one correction that was attached to the wrong verdict;
- true statements paired with explanations of different steps;
- internal answer-reference and review-provenance phrases;
- machine-like step labels;
- five overlong planning targets;
- three duplicated or generic method plans.

Positive and negative atomic examples now come from the corrected and incorrect sides of the same reviewed misconception record. This makes the verdict and explanation address the same scientific or mathematical point.

## Processor checks

All 180 views were prepared with the pinned `mlx-community/Qwen3.5-4B-4bit` processor revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`. The local verification confirmed:

- every source image hash and image tensor is present;
- user and image-prompt tokens are excluded from the training loss;
- every assistant target decodes exactly to its intended JSON plus the chat terminator;
- all views remain within the 2,048-token preparation bound;
- no development-family overlap recorded by the project split registry;
- no training release file and zero optimizer updates.

These checks establish transport and target integrity. They do not show that the model has improved or independently certify every scientific solution.

## Next gate

The target remains at least 300 independent reviewed families. Expansion will prioritize Chemistry Advanced and multi-step visual problems. Before training, the project must freeze a family-aware sampler, prove one-update checkpoint recovery and adapted-weight export, and lock unchanged-4B comparison inputs and stop rules. Any adapter that loses preservation canaries or degrades answer, procedure, subject-cell or completion behavior is rejected while the unchanged base remains intact.
