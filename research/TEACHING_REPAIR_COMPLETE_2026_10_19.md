# Teaching-exercise repair queue completed

19 October 2026. **Data-quality milestone; no model training or new accuracy result.**

All 95 families in the teaching-target repair queue have been reworked across eight separately checked and closed batches. Their 285 linked exercises now have preserved before/after versions. Zero source questions were discarded during this repair round.

- 83 families have their recorded repair checks complete.
- 12 retain source, convention, or independent-review clearance requirements and remain excluded from training eligibility.
- 94 final-answer mappings were preserved; one incorrect printed-option mapping was corrected with source and calculation evidence.
- All 285 final repaired views passed image transport, assistant-only masking, exact target decoding and a 2,048-token limit in the pinned Qwen3.5-4B processor. No optimizer updates were performed.

| Repair batch | Families reworked | Repair checks complete | Additional clearance |
|---|---:|---:|---:|
| 01 | 12 | 11 | 1 |
| 02 | 12 | 11 | 1 |
| 03 | 12 | 11 | 1 |
| 04 | 12 | 11 | 1 |
| 05 | 12 | 11 | 1 |
| 06 | 12 | 11 | 1 |
| 07 | 12 | 7 | 5 |
| 08 | 11 | 10 | 1 |

## What changed in the teaching

Each source problem has a specific step-check exercise, an answer-free method plan, and a justified solve/commit exercise. The repairs address corrupted notation, irrelevant atomic checks, plans that revealed the answer, missing branches or bounds, inconsistent conventions and verification that merely appealed to an answer reference. Correct final answers were retained while their explanations were repaired.

Automated review was assessed against source images and selected exact calculations. Repair cross-review saw proposed targets and was not blind. Additional source-only checks were recorded separately; missing or timed-out responses were not counted as successful evidence. Some responses gave the right answer with a wrong extra explanation, which was rejected. This is machine-reviewed experimental material, not human-certified data.

## Full-pool status

The archive retains 300 independent source families and 900 linked views. A separate staged copy combines the 158 previously supported families with 83 checked repairs: **241 provisionally supported families**, with **59 excluded from eligibility** (12 residual repair cases plus 47 other original holds). Original review snapshots and active-candidate identity remain unchanged. The staged pool contains held examples and must not be used directly as a training release.

Collective checks verified 419 original inventory source hashes, 878 closed-batch evidence hashes, existing train-split reservations and all image paths. The full archive retains six subject/exam cells of 50 families and 150/150 atomic pass/fail labels. These are data checks, not a model-performance score.

## Next gate

Freeze a named balanced pilot from the supported pool, excluding all residual holds; validate its family grouping, split reservations, targets, images and masks collectively. A 120-family pilot with 20 families per subject/exam cell and equal atomic pass/fail counts is feasible from the current pool, but is not yet selected or released. Lock the unchanged-model comparison, validation inputs and accuracy/procedure/completion regression rules before a short resumable training run.

This work improves the candidate teaching data. It does not establish that fine-tuning will improve the solver; that requires a matched experiment. Earlier negative model results remain unchanged. Private questions, answers, image assets, evaluation content and raw review traces are excluded from this public report.
