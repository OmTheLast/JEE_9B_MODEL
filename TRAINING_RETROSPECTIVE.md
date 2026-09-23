# What the 9B training experiments taught us

23 September 2026. This retrospective separates measurements from explanations we still need to test. Its purpose is to keep the approximately4B program from repeating the9B mistakes. The original Qwen3.5-9B base and vision weights stayed frozen in every adapter experiment. Training never damaged or overwrote the base model: every regression described here came from a removable LoRA adapter.

## Executive conclusion

We successfully built recoverable training, checkpointing, paired inference and answer/procedure evaluation. We did **not** produce a reliably better JEE solver. The strongest selected9B adapter tied the unchanged model on a separate evaluation, later checkpoints sometimes collapsed into short malformed replies, and more families or wider adapter coverage did not fix generalization. The experiments were valuable because they identified what not to scale.

Fine-tuning did exactly what its objective requested: it made the model more likely to reproduce our assistant targets. That objective was only an imperfect proxy for solving new JEE questions. With few independent source problems, several derived task formats, long structured targets and aggressive/repeated updates, the adapter learned local response patterns more readily than a robust problem-solving policy.

## What was measured

These rows use different question sets and must not be combined as one accuracy curve.

| Study | Main measurement | What it established |
|---|---|---|
| First controlled learning | Validation9/12 unchanged→11/12 selected step60; separate evaluation22/29→22/29, valid procedures19→18 | A small selection-set gain did not generalize. Step60 was useful inside that experiment but was not a better general solver. |
| First run, later checkpoints | Validation step1203/12 and step1802/12; many late outputs were tiny JSON-like replies | More updates could severely damage answer behavior even while training completed normally. |
| Recipe02 representation/solution arms | New36-question development: unchanged19, Arm A19, Arm B20; valid procedures18,18,19 | Changing the mixture produced question-level gains and losses, but neither arm cleared the predeclared improvement gate. |
| Matched learning-rate probe | At step90, original `1e-5` arm made very short replies on5/6 train prompts; matched `5e-6` arm made0/6 | Update size contributed to the short-reply transition in that configuration. This did not prove better reasoning. |
| Reliable01 consistent targets | Base/steps30/60/90 all7/12; procedure points17/16/18/17; training loss fell | Lower rate and consistent output style prevented short collapse, but stable output was not improved problem solving. |
| Reliable02 wider coverage | Last16-layer arm8/12 versus base7/12, but procedure14/24 versus17/24 | More trainable capacity improved completion/one answer while making reasoning less defensible. |
| Reliable02 more families |60 families, one exposure each:7/12 and16/24 versus base7/12 and17/24 | Doubling families under the same target contract was insufficient. More of the same data was not the immediate cure. |
| Target audit | Median69% of target words occurred before `SOLUTION`;59/60 had more pre-solution text than solution text | The curriculum devoted most supervised text to setup and headings. This is descriptive evidence, not proof that verbosity caused regression. |

## Why training made the adapter worse

### 1. We optimized imitation, not verified JEE success

Supervised loss rewards predicting the supplied next tokens. It does not directly reward a correct final answer on a new problem, the first valid derivation, checking every multi-correct option, or stopping after verification. Training loss therefore fell while held-out accuracy stayed flat. This was not an optimizer malfunction; it was an objective/goal mismatch.

### 2. Independent problem diversity was much smaller than row count suggested

Early data expanded one source question into representation, plan, execution, repair, structured-solution and autonomous-answer exercises. Those are useful task views, but they do not become independent physics, chemistry or mathematics knowledge. Thirty source families with several derived rows still expose the model to only30 underlying situations. Correlated repetitions make memorizing formats and local patterns easier than learning transfer.

### 3. Training and use-time formats competed

In the first curriculum, only one of six exercise types matched the actual product request: see a complete question and solve it autonomously. Other targets taught intermediate JSON-like representations, plans, selected steps and error judgments. The model was asked to internalize several output contracts through a very small LoRA. Recipe02 also compared arms with unequal supervised-token totals, so the exercise-type effect was confounded by exposure. The later bare-JSON replies are consistent with task/output interference, although the experiments do not isolate a single cause.

### 4. The initial update size and repeated exposure were too aggressive for the data

The original constant learning rate was `1e-5`. A controlled repeat at `5e-6`, with the same starting adapter and first90 examples, delayed the short-reply failure. This is the strongest causal evidence we have about collapse. Continuing through repeated passes over small correlated data also moved question-specific errors around. The exact interaction between learning rate, repetition and target format remains unresolved.

### 5. Small validation slices rewarded unstable checkpoint luck

The first selected adapter won11/12 versus9/12 on a12-question validation set, then tied22/29 on a separate evaluation and had one fewer valid procedure. A two-question validation gain can result from a few repaired delivery failures while other questions regress. Reusing small development slices further makes them diagnostic rather than independent evidence. We selected by the frozen rule correctly; the lesson is that the selection set and gate were too weak for a broad claim.

### 6. Wider LoRA coverage amplified an inadequate curriculum

Adapting16 layers gave the small dataset more influence. It reduced caps and gained one strict answer, but central reasoning quality fell. More capacity is not inherently better; it lets the optimizer fit both useful and harmful target tendencies more strongly. Coverage should be expanded only after the data contract demonstrates transfer under a narrow adapter.

### 7. More families preserved the same behavioral problem

Moving from30 to60 families was a valid controlled test and failed to improve aggregate answers. The responses still reconsidered settled steps and failed to commit. This rules out “just double the current dataset” as the immediate solution. It does not show that300 well-balanced, differently authored families would be useless.

### 8. Source and target reliability limited the learning signal

We found mismatched archived keys, source-crop issues, implicit assumptions, diagram-reading failures and physics/chemistry sign-convention nuances. A correct model response can be marked wrong by a bad reference; a wrong worked target can directly teach an error. Coordinator/model review is useful triage, but high-risk records still need source verification, independent derivation and subject review.

### 9. Correctness, procedure and delivery are different axes

Some answers matched the key through invalid reasoning. Some traces contained the right value but never emitted the required final line. Others reasoned well until a token cap. Later collapsed replies were not merely a formatting problem: many bare values were wrong. We must record exact answer, first invalid step, branch coverage, answer commitment, caps/loops and resource cost separately.

## What we cannot honestly claim caused the regression

- **Knowledge graphs:** useful for curriculum indexing and dependencies, but no graph-versus-no-graph controlled learning result exists.
- **Target verbosity:** the69% pre-solution share is suspicious and informs the next test; it is not causal proof.
- **LoRA itself:** narrow LoRA made experiments reversible and measurable. The failure was a recipe/data/evaluation result, not evidence that all LoRA training fails.
- **Insufficient output tokens alone:** extra tokens sometimes rescue an incomplete solution, but they can also allow longer reconsideration loops.
- **Base-model corruption:** base hashes were unchanged. Removing the adapter restores the original model.
- **One single mechanism:** learning rate has matched evidence for the short-output mode, but general answer regression likely combines data scale, target mismatch, exposure, capacity and selection noise.

## Mistakes in our original decisions

1. We attempted a broad procedural curriculum with only30 independent problems.
2. We counted many derived exercises as training volume without giving independent-family count enough weight.
3. We used too many output/task contracts relative to the size of the adapter and source pool.
4. We started at `1e-5` and scheduled repeated exposure before measuring the collapse boundary.
5. We allowed a frozen180-update engineering schedule to continue beyond the early useful checkpoint. Preserving the later checkpoints taught us about collapse, but they should never have been expected to improve monotonically.
6. We relied on small checkpoint-selection sets; the first validation improvement was too fragile.
7. One Recipe02 comparison did not match supervised target-token exposure tightly enough for clean attribution.
8. Recipe02 selected step30 before either arm had seen all60 intended parents.
9. We broadened adapter coverage before proving that the target behavior itself transferred.
10. We did not begin the product-sized4B baseline early enough; too much work treated9B training as the central path rather than a recipe/reference investigation.

## What we did correctly and should preserve

- Base/vision weights stayed frozen and hash-verified; regressions are removable.
- Every substantive run used resumable adapter, optimizer, scheduler, RNG and data-cursor checkpoints.
- Inputs, prompts, model revisions and image hashes were matched across paired comparisons.
- Training, development and protected-test roles were separated; failed evaluation outputs were not recycled into training.
- Selection and collapse rules were frozen before seeing the relevant results.
- We inspected procedures rather than publishing answer score alone.
- Negative results, source defects and post hoc corrections were preserved instead of overwritten.
- No unsuccessful adapter was promoted as a reliable JEE solver.

## Safeguards for the4B program

### Evaluation before optimization

- Freeze a24-question source-reviewed pilot, then a120-family development benchmark.
- Baseline9B/current prompt,9B/harness,4B/current prompt and4B/the same feasible harness before training.
- Keep2026 for one final evaluation after the recipe is chosen.
- Use larger paired gates and subject/exam-cell safeguards; never promote on training loss or one small slice.

### Data before scale

- Prepare at least300 independent training families,50 per subject×exam cell.
- Track independent families as the primary diversity unit; derived views are secondary.
- Make the main target resemble product use: decisive representation, necessary conditions, calculation, one verification and final answer.
- Create repair, branch-completion, commitment and abstention targets only when they address a documented failure.
- Keep concept/method graphs outside the assistant answer unless a controlled test shows benefit.
- Verify source pixels, answer, convention, arithmetic, units, option coverage, repetition and token length before release.

### Controlled optimization

- Start with one exposure per family, a conservative rate and narrow LoRA coverage.
- Checkpoint at25/50/75/100% of independent-family exposure and stop on answer/procedure/delivery regression.
- Match parent order and supervised-token exposure when comparing curricula.
- Change one factor at a time: target behavior, learning rate, coverage or harness component.
- Compare trained4B against its unchanged base **and** against9B+harness.

### Verified distillation and deployment

- Treat9B and other teachers as proposal generators, not sources of truth.
- Accept successful traces only after source-key, symbolic/numeric, unit/convention and branch checks; use human/subject review for high-risk cases.
- Preserve disputed traces as repair examples only after adjudication.
- Measure4B memory, load time, tokens/second and browser/server behavior. Roughly4-bit4B weights alone are about2GB before runtime and KV-cache overhead; plan a server fallback and later smaller student if needed.

## Final lesson

The failure was not that the model “needed more training.” The failure was that our training signal was too small, too correlated and not aligned tightly enough with verified end-to-end solving. More optimizer steps made the adapter better at the supplied distribution and worse at the behavior we actually cared about. The4B project should invest first in independent data, explicit verification, a realistic harness and a credible benchmark; optimization comes after those assets exist.
