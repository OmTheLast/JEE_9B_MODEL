# Solve→verify→commit harness contract v1

24 September 2026. This specification separates reasoning from answer serialization because the same frozen drafts improved when the final state was externally constrained.

## State 1: solve

Input: original image, visible response contract, subject hint only when supplied by the user.

Output ledger:

- representation and requested unknown;
- selected method and assumptions/convention;
- at most six active subgoals or option branches;
- calculations tied to named quantities/equations;
- one candidate result and unresolved items.

The solver may not re-open a completed subgoal without recording a failed check. Exact or near-exact repeated spans and unchanged recalculation trigger transition to verification instead of a larger token budget.

## State 2: targeted verification

The controller chooses only checks relevant to the response type and ledger:

- numeric: decisive equation, unit/scale, rounding and sign/convention;
- single-select: verify the chosen option and any materially unresolved competitor;
- multi-select/matching: mark every option/row pass or fail;
- diagram/structure: restate the critical visual relation and uncertainty;
- mathematics: symbolic substitution/equality, determinant/rank or boundary case;
- physics/chemistry overlap: state the subject convention before applying signs.

Verification returns bounded structured fields: checked claim, method, pass/fail/uncertain, correction if any, remaining uncertainty and proposed final value. It may not restart the complete solution. If a material image fact remains uncertain, route to visual assistance or abstain rather than inventing it.

## State 3: commit

The assistant response is prefilled at `FINAL_ANSWER:`. A constrained decoder or deterministic normalizer accepts only the visible schema:

- options: `{"type":"options","selected_options":["label"]}`;
- numeric: `{"type":"numeric","value":"decimal string"}`;
- unresolved: `{"type":"abstain"}`.

Commit cannot add reasoning or change an answer without a verification correction. The displayed student explanation comes from the bounded solve/verify ledger, not from another free-running generation.

## Component measurements

Measure each state separately: solve procedure, verification correction/false-change rate, commit parseability, strict answer, loops/caps, visual critical facts, latency and memory. Compare unchanged4B and unchanged9B under identical controller rules before training. A component is retained only when it helps on its trigger set without regressing previously correct cases.

## Current status

Commit behavior passed a development diagnostic after deterministic normalization. Natural-language answer commitment failed.

The first three-question Chemistry Advanced diagnostic rejected draft-conditioned verification:4B scored0/3 and9B obtained one correct label with scientifically invalid reasoning. A blind whole-question fact table also remained verbose. One independent claim per call produced24/24 parseable, uncapped outputs across4B and9B, but only1/12 and3/12 decisive chemistry claims were correct. Therefore atomic verification is retained as a stopping/format primitive, not a correctness oracle. High-risk checks should be independent of the solver's selected answer, compared externally, and route uncertainty to abstention or trusted evidence. Full evidence is in `VERIFICATION_DIAGNOSTICS_RESULT.md`.

Solve and scientific verification remain experimental. This file does not authorize training.
