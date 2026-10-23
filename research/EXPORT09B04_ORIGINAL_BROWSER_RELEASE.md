# Original 9B browser release — conversion verified

23 October 2026. The original-weight conversion passes full synthetic CPU and Chrome comparisons. Public upload and deployment receipts determine live availability; this report does not claim a general JEE improvement.

## What changed

The first prototype started from upstream BF16 weights, folded in an adapter and compressed again. That changed one synthetic image output. This release preserves the original MLX affine 4-bit codes, group-64 scales and floating offsets throughout the decoder, embeddings and output head. LoRA remains a separate rank-4, scale-2 calculation with explicit inputs; the unchanged base feeds zeros.

The custom runtime uses ONNX Runtime WebGPU and the Transformers.js tokenizer. It is not a standard Transformers.js `from_pretrained` model. Visitors download static files and compute in their browser, without an inference API or local server.

## Checks completed

| Check | Result | Scope |
|---|---|---|
| Full CPU generated sequences | 18/18 exact matches | Base plus five adapters, three synthetic fixtures each; matched-FP32 original MLX controls |
| Full Chrome generated sequences | 10/10 exact matches | Text for all six choices; two images for base and Reliable01 step 60 |
| Browser image preprocessing | Passed | Exact IDs/positions; maximum pixel error 5.91e-8 |
| Complete decoder layers29/31 | Passed | Outputs and recurrent/cache states over real 179-token inputs, matched FP32 |
| Published candidate files | 312 artifact hashes verified locally | 7,540,332,526 bytes, including optional vision and tokenizer |
| Harness/helper tests | 7 passed | Existing format/sign checks, embedding unpacking, image positions, loop guard |

Browser text runs took approximately 5.5–5.7seconds after loading for a 34-token prompt and 7 output tokens. Image runs process 179 prompt tokens. Download/loading/vision time is excluded from these decoder timings. The machine has 64 GiB unified memory; this does not establish minimum requirements or phone support.

## Original image failures remain

Images of `2 + 3 = ?` and `8 - 1 = ?` produce 4 and 2 instead of 5 and 7 in both original and faithful converted models. Matching those failures proves only consistency on these fixtures. Image recognition is experimental and unreliable; do not interpret this release as a reliable screenshot solver. Fixed 384×384 white-letterboxed inputs also limit dense diagrams and pages.

## Available archive selections

- Unchanged original 9B base.
- Reliable01:30 families, final 3 language layers, steps 30/60/90.
- Coverage:30 families, final 16 layers, step 60.
- Expanded data:60 families, final 3 layers, step 60.

A family is one independent source question with related teaching exercises. A step is one optimizer update. These are five archived snapshots, not every historical step. No adapter has been promoted as a generally better JEE solver.

## Implementation findings and retained failures

- Preserve already 1-centered, rounded MLX RMS normalization weights; use the correct convention in the target operation.
- Match MLX recurrent RMS normalization epsilon after converting a mean into a sum.
- Honor tokenizer end-of-turn248046 as well as configuration end 248044. Otherwise the converted model can continue after the original stops.
- Keep optional vision FP32: the earlier FP16 browser comparison failed.
- An initial Chrome run was interrupted by the development server reloading after an HTML edit. It was excluded; the recorded completed run was restarted cleanly. Keep verification pages isolated from live source edits.
- No training was performed. The original MLX archive and failed conversion experiments remain available.

## Runtime limits

About 5.69 GB text files and adapter snapshots; optional image encoder adds 1.84 GB. Stop/reset terminates the worker; repetition detection, output budget and 10-minute decoder limit bound generation. Storage caching is optional and may be evicted. Questions stay in the browser. The model can produce confident errors and incomplete answers.

[Browser source](../web/) · [Original MLX archive](https://huggingface.co/OmTheLast/jee-solver-experimental-mlx-adapters) · [Browser weights](https://huggingface.co/OmTheLast/LlamaJEE-9B-ONNX-WebGPU) · [Project site](https://ompatnaik.com/llamajee/).


## Checkpoint activation check

A further six Chrome text runs also compared the first generated token score with the saved native ONNX controls. Each adapter changes that score from the unchanged base, and every browser score matches its corresponding control within 0.000025. This verifies that the checkpoint selector changes numerical calculations even when all six models give the same final answer. The score is not answer confidence or an accuracy measure. The real harness also completed base and coverage-adapter text runs; stop/reset and the retained 0.8B selector were checked.


## Hosted model artifact

Pinned model revision: `81f9f6a2591a348994e655a8ea8302c392b9dce2`. All315 uploaded file contents were verified using227 LFS SHA256 and88 Git blob hashes. The website uses this immutable revision. Public-site generation is verified separately after deployment.
