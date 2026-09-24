# Verification diagnostics before 4B training

24 September 2026. Three reused Chemistry Advanced failures were used only to test solver-control mechanisms. They are not training examples and do not estimate general JEE performance.

## What was tested

All calls used unchanged weights, original question images, deterministic decoding and no answer key.

1. **Draft-conditioned verifier:** read the failed 4B draft and checked up to four claims before proposing an answer.
2. **Blind fact table:** did not see the draft or proposed answer, but extracted all important facts in one response.
3. **Atomic blind checks:** saw only the original image and one narrowly specified chemistry claim per call. It did not choose an answer.

## Result

| Condition | Qwen3.5-4B | unchanged 9B control | Meaning |
|---|---:|---:|---|
| Draft-conditioned strict answers | 0/3 | 1/3 | The 9B correct label had invalid reasoning. |
| Draft-conditioned valid procedures | 0/3 | 0/3 | Both models largely rationalized incorrect chemistry. |
| Draft-conditioned full JSON | 1/3 | 2/3 | The supposedly bounded state still capped. |
| Blind fact-table full JSON | 1/3 | 0/3 | Removing the old answer did not control response length. |
| Atomic-check full JSON | 12/12 | 12/12 | One claim per call fixed stopping and format. |
| Atomic decisive chemistry claims | 1/12 | 3/12 | Small calls did not fix missing or confused chemistry. |

The 9B control correctly traced three of five terminal products in one named-reaction matching question. The 4B model correctly assigned only the aluminium group reagent among all twelve checks. Both models confidently reversed qualitative-analysis media and misread the phthalimide/Gabriel sequence.

## What this changes

The harness can solve **mechanical** failures: it can stop loops, force a parseable response and isolate one claim. It cannot manufacture chemistry knowledge. A verifier that receives a wrong draft is also vulnerable to confirmation bias. Therefore:

- keep independent checks blind to the solver's selected answer where possible;
- use one claim per call for high-risk reaction, structure and reagent checks;
- compare the independent finding to the solver ledger externally;
- abstain or route unresolved claims instead of letting a second fluent answer override the first;
- teach the missing rules through independent analogous families, never these evaluation questions.

## Training implication

Training remains locked. The next dataset release must contain concise, consistent exercises at three levels:

1. **atomic rule:** identify one endpoint, reagent condition, sign convention or structural fact;
2. **composition:** connect several verified facts into a method or reaction chain;
3. **solve and commit:** solve a complete independent JEE problem, verify the decisive claim once and emit the exact answer schema.

Every family needs source and answer review, and every subject/exam cell needs coverage. The initial last-three-layer rank-4 adapter remains conservative. Checkpoints can continue only when they preserve the unchanged 4B canaries and do not lose strict answer or procedure score. A regressed adapter is rejected; the immutable base remains deployable.

## Decision

Do not increase verifier tokens and do not start optimizer updates. Build the first reviewed training-data tranche and method registry around independent analogues of the observed error classes, while preserving the 300-family and independent-confirmation gates in `TRAINING_SAFETY_GATES_V1.md`.
