# Selecting the small JEE model: staged benchmark plan

23 September 2026. We should benchmark the three user-proposed families, plus narrowly chosen challengers, but should not run a large protected benchmark on every available model. Selection uses a funnel and hard product gates.

## Candidate roles

| Candidate | Why it is here | Main concern before testing |
|---|---|---|
| Qwen3.5-4B | Same family as the9B reference; Apache-2.0; official multimodal HF, MLX4-bit and ONNX artifacts exist | Browser speed and a reproducible path from our trained weights to the community-style browser graph; the Qwen3.5 Transformers.js exporter is not currently public |
| Gemma3n E4B-it | Designed for low-resource devices; native multimodal; Transformers.js supports Gemma3n | It is roughly8B total parameters, gated by Gemma terms, and current MLX-VLM LoRA docs exclude Gemma3n |
| LFM2.5-VL-3B | Official native, MLX and Transformers.js ONNX builds; strong OCR/deployment story | Vendor says it is not recommended for reasoning-intensive technical work |
| Qwen3-VL-4B-Thinking | Reasoning-oriented smoke-screen challenger | No matching maintained browser Thinking build confirmed yet |
| Phi-4 multimodal | Optional server challenger if all small browser-first models reason poorly | Larger/custom-code and no confirmed direct Transformers.js path |

The exact revisions and current facts are frozen in [`candidate-manifest-v1.json`](candidate-manifest-v1.json). This is a runtime inventory, not a benchmark result.

## Stage0 — non-negotiable gates

Before a model sees the JEE pilot it must pass:

1. **Identity:** exact model/revision, instruct rather than base checkpoint, image input and processor recorded.
2. **License:** redistribution and derivative deployment acceptable. Gemma access/terms and LFM1.0 need explicit review; Apache/MIT still require attribution.
3. **Native run:** one text and one image fixture complete locally with deterministic settings.
4. **Training path:** a one-update LoRA preflight changes only intended tensors and resumes; no broad training.
5. **Deployment path:** a maintained ONNX/Transformers.js build is enough to enter the unchanged-base smoke, but the eventual training winner also needs a reproducible adapted-weight path: exporter, or verified weight injection into a matching full-precision ONNX graph followed by quantization. A community base ONNX file alone cannot deploy our LoRA.
6. **Resource sanity:** estimated download below4GB for the chosen browser variant and measured peak below the target-device limit. These are initial engineering limits, not claims of phone universality.

A model can remain a server/reference challenger after failing the browser gate, but cannot win the primary browser-model decision.

## Stage1 — six-question smoke screen

Use six source-reviewed **development-only** questions, one per subject × Main/Advanced cell. They must not be the protected final set and must never later enter training. Include:

- one dense equation/table image;
- one geometry/physics diagram;
- one organic/inorganic structure or reaction scheme;
- one cross-subject convention case;
- single-choice, multiple-choice and numeric response contracts;
- at least two questions that require more than one dependent step.

Run the same image, concise product prompt, deterministic decoding,4,096-token ceiling and no tools. Record strict answer, parseable final, procedure0/1/2, cap/loop, elapsed time, prompt/generated tokens and peak native memory. Drop a candidate only for a hard failure: runtime/image incompatibility,0/6 with unusable procedure, repeated corruption/loops, or failed deployment/license gate. The six items choose who deserves the full pilot; they are not an accuracy claim.

## Stage2 — frozen24-question model-selection pilot

Use24 independent, source-reviewed development families, four per mathematics/physics/chemistry × Main/Advanced cell. Freeze images, gold answers, answer contracts, conventions and scorer fixtures before any candidate call.

### Coverage

- Mathematics: algebra/matrices, calculus, coordinate/geometry, probability or sequences; at least two graphs/diagrams.
- Physics: mechanics, electricity/magnetism, optics/waves and thermal/modern; at least two diagram-heavy and one convention-sensitive problem.
- Chemistry: physical, inorganic and organic; include a table, a reaction/structure image and physics/chemistry thermodynamics or atomic overlap.
- Eight mostly textual/equation images, eight visual diagrams/graphs/structures and eight mixed tables/layouts.
- Eight single-choice, eight multiple-choice and eight numeric/match-style contracts where source availability permits.

### Matched conditions

Every candidate receives the original image, identical product-level instructions, no gold transcription, no answer hint, no tools, deterministic decoding and the same4,096-token maximum. Native processors may resize/tokenize according to the model card; record the resulting image/token dimensions. Then rerun only the finalists with the fixed harness. Never compare a harnessed candidate with an unassisted candidate as the training effect.

### Primary measurements

1. Strict final answer, reported overall and separately for all six cells.
2. Procedure validity:0 invalid,1 partial,2 valid; central algebra/science errors override a lucky final answer.
3. Completion: parseable final, cap, exact repetition, unsupported abstention and option/branch coverage.
4. Visual fidelity: critical values/symbols/units, option labels, table cells and diagram relations recovered correctly.
5. Runtime: time to first token, total time, generated tokens/s, peak memory and browser download bytes.
6. Deployability: clean-cache browser load, WebGPU failure behavior, native-versus-ONNX answer agreement and server path.
7. Trainability: one-update memory/time and exact adapter/export support. Training loss is not a selection score.

## Decision rule

Use hard gates before ranking:

- no subject/exam cell may score0/4;
- at least18/24 parseable final answers;
- no repeated short-output/corruption mode and at most6 token caps;
- no more than one critical visual transcription failure in the visual audit subset;
- a working or demonstrated reproducible ONNX/Transformers.js path;
- measured browser peak/download within the declared test-device budget;
- acceptable license and a resumable LoRA path.

Among models that pass, choose the Pareto winner rather than hiding tradeoffs in one arbitrary score. Primary order is strict answers, then procedure points, then visual fidelity. If two models differ by no more than2/24 answers and3/48 procedure points, prefer lower browser memory/latency and simpler export. Keep the other as a server or fallback candidate if it has a meaningful cell advantage.

On the later120-question benchmark, retain the existing rule: the small model should be within about10 percentage points of9B+harness and have no catastrophic cell before it becomes the primary fine-tuning target. The24-item pilot chooses what to scale; it does not establish general JEE reliability.

## Stage3 — browser parity and training choice

For the top two, compare native and ONNX builds on six public/synthetic fixtures plus the24-item outputs without opening protected final data. Require identical preprocessing contracts, compatible first-token/selected logits within a declared tolerance where accessible, and no answer/procedure regression larger than one item after quantization. Then perform one isolated LoRA update and recovery test. Only after those gates do we author/train the first30 concise families.

## Immediate order

1. Complete the current9B adapter-format bridge and learn where conversion breaks. As of23 September, public Optimum-ONNX support lists Qwen3 but not Qwen3.5, and the Transformers.js Qwen3.5 exporter request remains open; treat that as a measured deployment risk rather than assuming the community ONNX artifact is reproducible.
2. Run clean native/browser smoke tests for Qwen4B, Gemma E4B and LFM3B; add Qwen3-VL Thinking only if its deployment gap is tractable.
3. Freeze the six-question screen and24-question manifest without using2026 final.
4. Screen candidates; run the24-question pilot on survivors.
5. Select top two for ONNX parity and one-update training preflight.
6. Select the primary4B-class model, publish the decision evidence, then begin reviewed training data.
