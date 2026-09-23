# Completion-control diagnostics: what worked and what did not

23 September 2026. Both tests used unchanged Qwen3.5-4B and unchanged Qwen3.5-9B weights on their own capped outputs from the reused24-question development pilot. No gold answer, reviewer note or training update entered either pass.

## 1. Natural-language answer controller failed

The model received its unfinished draft and original image, with a request to check the decisive step and commit one JSON answer within512 new tokens.

| Model | Triggered caps | Controller caps | Controller correct | Effective strict score | Effective parseable finals |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 11 | 9 | 0 | 13/24 → 13/24 | 14/24 → 15/24 |
| Qwen3.5-9B | 8 | 7 | 0 | 15/24 → 15/24 | 16/24 → 17/24 |

Another request to be concise caused another round of reasoning. It did not solve the stopping problem.

## 2. Forced answer serializer helped

The second diagnostic pre-filled the assistant response with `FINAL_ANSWER:` and allowed96 tokens to produce only the answer object. It stopped normally on all11 4B items and on six of eight 9B items. Models still used inconsistent schemas (`{"option":"B"}`, bare `C`, `{"answer":"A"}`), so a deterministic normalizer converted only explicit choices/numbers using the visible response type and option labels. It did not consult the answer key.

| Model | Baseline strict | Trigger-only serializer, effective strict | Baseline parseable | Normalized effective parseable |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 13/24 | **15/24** | 14/24 | **24/24** |
| Qwen3.5-9B | 15/24 | **19/24** | 16/24 | **24/24** |

![Raw versus completion-controlled strict answers](../charts/completion_control_pilot24.png)

The 4B recoveries were the osmotic-pressure numeric answer and a sheet-field option. The 9B recoveries were named-reaction matching, coaxial flux, a logic circuit and a sheet-field option. Coordinator review found the preceding drafts clearly supported three of the six recoveries and partially supported the other three; none was an unsupported lucky answer. Independent expert review remains pending.

## Interpretation

This is a harness gain, not a model-weight or training gain. The first pass often contained enough work to answer but kept rechecking until its token limit. Natural-language instructions did not reliably stop it. Constraining the output state and normalizing a small answer grammar did.

The serializer also produced wrong, confidently formatted choices on unresolved drafts. Therefore it improves answer availability, not scientific reliability by itself. A useful product needs three distinct states:

1. **solve:** bounded decomposition and calculation;
2. **verify:** targeted checks tied to a claimed result, with explicit uncertainty;
3. **commit:** grammar-constrained answer serialization that cannot restart the solution.

The next model-selection comparison should use this state machine for both4B and9B. First run the serializer path on all24 only as a trigger policy test and finish coordinator procedure review. Then add bounded subgoal/branch checks for the remaining wrong drafts, especially Chemistry Advanced. Do not train the serializer behavior into the model until we know which failures are reasoning versus state control.

## Decision

- Qwen3.5-4B remains the primary small-model candidate; with this harness it matches the raw9B baseline at15/24 on reused development data.
- The harnessed unchanged9B result19/24 becomes the more appropriate reference for the later4B target; the raw15/24 score understated available reasoning.
- Neither result is a promotion or broad benchmark claim. The questions are reused, procedure review is incomplete and Chemistry Advanced remains too weak.
- A separate visual frontend remains a later targeted ablation. Completion control produced more immediate gain than adding a second visual model can currently justify.
