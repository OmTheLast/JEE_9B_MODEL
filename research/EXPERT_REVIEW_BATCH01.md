# First risk-based review batch

27 September 2026.

> **Historical stage:** This report records the original two-reviewer result. The held family was later quarantined and replaced under the [machine-review fallback](MACHINE_REVIEW_BATCH01_RESULT.md).

## Status

The first independent review batch for the 300-family 4B curriculum candidate is complete. Twelve high-risk families were selected before reviewer output, with two from each Mathematics, Physics and Chemistry × Main and Advanced cell. Two model reviewers independently inspected the original question images without candidate answers, solution targets, answer keys or each other's output.

This is review evidence, not human subject-expert certification. Every family remains `training_ready:false`, and no optimizer update or solver evaluation ran.

## Result

| Outcome | Families |
| --- | ---: |
| Candidate supported by both blind reviewers | 10 |
| Candidate correction required | 1 |
| Reviewer disagreement held | 1 |
| Human subject-expert pending | 12 |

The correction was substantive: an active chemistry target omitted one graph branch and therefore taught the wrong final choice. The prior batch artifacts were archived, the scientific derivation and answer were corrected, and all three linked atomic-check, method-composition and solve/commit views were rebuilt.

The disagreement concerns a physics convention. Both reviewers agree on the wavelength, speed and harmonic classification, but disagree on whether the stated end-correction magnitude is valid under the problem's sign convention. That family was held. We did not resolve it by majority vote.

## Verification after the correction

- The corrected 12-family component retains 36 linked views and its authoring replay is byte-stable.
- Its structural audit passes with six atomic pass and six fail targets.
- The pinned Qwen3.5-4B processor verifies all 36 images, exact decoded targets and assistant-only masks.
- Corrected-component views span 255–982 total tokens; none exceeds the 2,048-token limit.
- The collective candidate audit still passes 300 independent families and 900 linked views.
- Post-correction candidate-index SHA256: `6e2cfda820cc9abf89bbb10a38fcea1d997e886d222688f1ffed560bfb1582b4`.
- Optimizer updates: 0.

Private questions, images, answers, derivations and reviewer transcripts are excluded from this repository.

## Process failures that mattered

The review workflow itself exposed four controls we now require:

1. **A parsed JSON object is insufficient.** One reviewer used wrong JSON value types. Its first file was preserved, and a reviewer-issued schema-correct version was validated independently.
2. **Scientific answers and printed response labels are separate fields.** Nested multiple-choice questions can have statement letters inside a numbered outer choice.
3. **Local reviewer concurrency is unsafe.** One parallel call hit the OpenCode database lock; the failed attempt was retained and the runner now defaults to one worker.
4. **A frozen review batch must remain immutable.** Recomputing risk scores after a correction briefly changed which items would be selected. The original batch hash was restored, and the builder now validates and preserves completed batch inputs.

## Next gate

1. Obtain human physics adjudication for the convention disagreement.
2. Human spot-check the supported high-risk pool.
3. Continue high and medium risk reviews in immutable batches, holding every transcription, scientific, convention or option-mapping discrepancy.
4. Freeze a separately hashed training release only after human/discrepancy adjudication.

The unchanged 4B base remains the fallback. Model-review agreement alone cannot authorize training data.
