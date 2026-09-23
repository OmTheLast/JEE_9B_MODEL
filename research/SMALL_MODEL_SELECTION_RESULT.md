# Small-model selection: runtime screen and 24-question pilot

23 September 2026. This report records the first unchanged-model selection pass for a browser/phone JEE solver. It is development evidence, not an official benchmark and not a claim of general JEE reliability.

## Decision

**No small model is selected for training yet.** Qwen3.5-4B is the only current survivor. It is close enough to the unchanged Qwen3.5-9B control overall to justify further work, but it fails the predeclared completion and Chemistry Advanced gates.

The immediate next experiment changes the solving harness, not the weights: run a matched concise answer-commit/loop-stop condition on Qwen3.5-4B and the unchanged 9B, first on the capped and Chemistry Advanced items. If it reduces caps without harming correct answers or procedures, repeat it across all 24. Only then perform the one-update LoRA/export preflight and choose whether to train.

## Why the candidate list is open

The user proposed Qwen3.5-4B, Gemma 3n E4B and LFM2.5 and explicitly allowed stronger alternatives. Candidates are admitted by product gates rather than brand or parameter label:

1. native image and text inference works;
2. license and redistribution are acceptable;
3. JEE reasoning and final-answer behavior survive a six-item screen;
4. a maintained Transformers.js/ONNX runtime exists or can be reproduced for trained weights;
5. browser download, memory and latency fit the device target.

Qwen3-VL-4B-Thinking was added as a reasoning challenger. Cohere Labs' Apache-2.0 North Micro Vision 2.4B is on the watchlist because it is compact and vision-capable, but it has no confirmed matching Transformers.js ONNX path, so it cannot currently win the browser-primary role. Phi-4 multimodal remains a possible server-side challenger. We will not download and fully test every small VLM: a model must have a plausible advantage and deployment path before it consumes the frozen JEE pilot.

## Stage 0: runtime and deployment gates

| Candidate | Native synthetic text/image | Native resource observation | Browser/export status | Outcome |
|---|---|---|---|---|
| Qwen3.5-4B | Passed; both answered 9 m | 0.87 s load; 3.31 GiB text and 3.98 GiB image reported peak | Exact community q4f16 bundle 2.80 GiB; clean WebGPU load and image generation passed. Trained-weight Qwen3.5 exporter remains unavailable in current public tooling | Advanced |
| LFM2.5-VL-3B | Passed; both answered 9 m | 0.73 s load; 3.89 GiB text and 4.34 GiB image reported peak | Official ONNX/browser artifact exists; selected bundle is 3.71 GiB | Advanced to six-item screen, then rejected |
| Gemma 3n E4B-it | Not run | Repository access requires accepting Google Gemma terms; it is about 8B total parameters despite the E4B active-memory name | Transformers.js architecture support exists, but exact artifacts remain inaccessible to this account | Blocked, not scored |
| Qwen3-VL-4B-Thinking | Native JEE screen ran | Exact MLX snapshot verified | No matching maintained browser Thinking build confirmed | Rejected after screen |

The Qwen browser base proves that the browser runtime can load and generate. It does not prove that our future trained adapter can be exported. The separate 9B conversion experiment mapped the MLX adapter exactly to PEFT and passed BF16/native text-image parity, but stopped because current public Qwen3.5 Optimum/Transformers.js exporter support is incomplete. A community ONNX base is not a reproducible trained-weight pipeline.

## Stage 1: six-question screen

The screen reused six coordinator-reviewed 2025 development questions, one per subject × exam cell. All candidates received the same images, prompt, deterministic decoding and 4,096-token ceiling.

| Candidate | Strict answers | Parseable finals | Caps / malformed replies | Procedure points | Decision |
|---|---:|---:|---:|---:|---|
| Qwen3.5-4B | 3/6 | 4/6 | 3 caps | 6/12 | Conditional survivor; completion control required |
| LFM2.5-VL-3B | 0/6 | 0/6 | 5 short malformed/echo replies | 0/12 | Rejected |
| Qwen3-VL-4B-Thinking | 0/6 | 0/6 | 6 caps | 2/12 | Rejected |

Setting `enable_thinking=false` on the Qwen3-VL Thinking checkpoint produced byte-for-byte identical outputs on all six questions. In this checkpoint/template/runtime combination it did not provide a usable concise mode.

LFM's quick synthetic answer was therefore not evidence of JEE reasoning. The real screen rejected it before a roughly 3.7 GiB browser download was attempted.

## Stage 2: matched 24-question pilot

The pilot combines two already reviewed 2025 development releases. It contains 24 unique families, four per mathematics/physics/chemistry × Main/Advanced cell. These questions are now consumed development/model-selection data. They cannot later serve as untouched evaluation.

| Condition | Strict answers | Parseable finals | 4,096-token caps |
|---|---:|---:|---:|
| Qwen3.5-4B | **13/24** | 14/24 | 11/24 |
| Unchanged Qwen3.5-9B control | **15/24** | 16/24 | 8/24 |

The 4B is two answers, or 8.33 percentage points, below the 9B on this set. That falls within the rough ten-point continuation boundary. It does not pass selection because the predeclared hard gates also protect weak cells and unusable outputs.

| Cell | Qwen 4B | Unchanged 9B | Qwen 4B caps | 9B caps |
|---|---:|---:|---:|---:|
| Chemistry Advanced | **0/4** | 3/4 | 4/4 | 1/4 |
| Chemistry Main | 3/4 | 3/4 | 1/4 | 1/4 |
| Mathematics Advanced | 2/4 | 2/4 | 2/4 | 2/4 |
| Mathematics Main | 3/4 | 3/4 | 1/4 | 1/4 |
| Physics Advanced | 2/4 | 2/4 | 2/4 | 2/4 |
| Physics Main | 3/4 | 2/4 | 1/4 | 1/4 |

The 4B gained three items relative to 9B and lost five; both got ten right and both missed six. This is not a simple “smaller model is uniformly worse” result. Its clear localized failure is Chemistry Advanced, where every output hit the token ceiling and only one final was parseable. The 9B also fails the product completion gates, so size alone does not solve answer commitment.

![Pilot comparison](../charts/model_selection_pilot24.png)

## Separate small visual model: objective decision

A small visual frontend could help with crop validation, table transcription, chemical labels, circuit connections and explicit uncertainty. There is evidence for a targeted test: Qwen4B solved some visual questions, including a graph/area and Vernier item, but misread a logic-gate connection and an organic structure.

It should **not** be the default architecture now:

- the largest measured problem is completion and Chemistry Advanced reasoning, which an OCR/VLM helper will not fix;
- a text-only visual handoff can lose geometry, arrows, graph curvature, stereochemistry and superscripts;
- the first candidate helper adds about 770 MB plus load/latency to an already multi-gigabyte browser model;
- every main solver candidate already accepts the original image.

Keep the original image as solver input. Later compare integrated vision, transcription-only and hybrid original-image-plus-transcription on 12 visual development questions. Retain a helper only if hybrid use produces a measured end-to-end gain under the existing no-regression and device-resource rule. Route it to categories where it helps instead of loading it for every question.

## What this means for training

The pilot does not say “train Qwen4B now.” It says Qwen4B is promising enough to diagnose. Training before fixing the evaluation and completion behavior would repeat the earlier 9B mistake: adapting a model to a narrow target while the product failure may come from prompting, stopping and answer commitment.

The next order is:

1. Freeze a concise answer-commit/loop-stop harness variant.
2. Test it on the 11 Qwen4B caps and all Chemistry Advanced cases, with an unchanged 9B control.
3. If it helps without damaging correct procedures, rerun all 24 under matched conditions.
4. Require a reproducible trained-weight browser path and one-update recovery/export preflight.
5. Select the model, then train on independent concise procedural families. Do not train on these 24 development questions.
6. Expand to the 120-question development benchmark and retain a sealed final set for the eventual claim.

## Follow-up completion diagnostic

The immediate diagnostic was completed after the pilot. A second natural-language request to be concise failed: the4B stayed13/24 and the9B15/24, with16 of19 controller calls hitting their shorter512-token ceiling. A narrower assistant-prefill serializer plus deterministic visible-label normalization recovered supported answers: provisional effective scores became **15/24 for4B** and **19/24 for9B**, with24/24 parseable finals in both trigger-only product results. See [Completion-control diagnostics](COMPLETION_CONTROL_RESULT.md).

This makes the next architecture clearer: separate bounded solving, targeted verification and grammar-constrained answer commitment. It does not remove the Chemistry Advanced reasoning problem and it is not a training gain.

## Evidence boundaries

- The 24 questions are reused development data, not an untouched, official or statistically precise JEE benchmark.
- Full 24-item coordinator procedure review remains pending; current Stage 2 numbers are strict answer/format/completion measures.
- Public-paper presence in base pretraining cannot be audited.
- Quantized 4B and 9B artifacts are a product-behavior comparison, not a pure architecture experiment.
- Gemma was blocked by access/terms, not defeated by a score.

## Official model/runtime references

- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Gemma 3n E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)
- [LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) and [official ONNX build](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-ONNX)
- [Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)
- [North Micro Vision](https://huggingface.co/blog/CohereLabs/meet-north-micro-vision-instruct)
- [Transformers.js documentation](https://huggingface.co/docs/transformers.js/en/index)
- Open Qwen3.5 export gaps: [Transformers.js issue 1574](https://github.com/huggingface/transformers.js/issues/1574) and [Optimum-ONNX issue 131](https://github.com/huggingface/optimum-onnx/issues/131)
