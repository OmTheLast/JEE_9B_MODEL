# Actual 9B browser prototype — 23 October 2026

A custom exporter now runs the complete text path of the pinned Qwen3.5-9B with the converted **Reliable01 step-60 adapter** in Chrome WebGPU. This is one unselected historical checkpoint, not a promoted JEE solver. No training was performed.

## Verified

- Exported all 32 decoder layers, the final output projections and token embeddings. The text package is 6,999,102,442 bytes (about 7.00 GB).
- The real trained recurrent and full-attention layer graphs match the Transformers reference on three synthetic state inputs each. Adapter effects were nonzero before quantization.
- The browser and CPU versions of the full quantized text model generated **exactly the same 271 tokens**, stopped at EOS and finished with `FINAL_ANSWER: 9` on a synthetic constant-acceleration problem.
- Chrome run: **71.85 seconds**, including model loading. This is a local static-file test on a 64 GiB Mac, not internet download timing, phone performance or a measured browser memory peak.
- Exported the actual vision encoder for a fixed 384 × 384 input. Full-precision browser features matched the reference with relative L2 error about 0.0000226. The vision file adds about 1.84 GB.

The browser performed inference itself. The localhost process served static model files; it did not compute answers. The runner uses ONNX Runtime WebGPU directly and Transformers.js tokenization. It is **not** a drop-in package for the standard Transformers.js Qwen model loader.

## Failed checks preserved

The synthetic image visibly says **2 + 3 = ?**, whose answer is 5:

| Runtime | Unchanged base | Reliable01 step 60 |
| --- | --- | --- |
| Original MLX | 4 | 4 |
| Original upstream BF16 / PEFT | 4 | 4 |
| New quantized ONNX CPU | Not tested | 6 |
| New quantized browser path | Not tested | 6 |

The original image-recognition failure predates this export. The portable output also changes; quantization, precision and shared preprocessing compatibility need isolation before claiming image-answer fidelity. These results **do not show that fine-tuning caused the image failure**. Both ONNX runtimes agreed token-for-token on their wrong image answer.

The FP16 vision graph failed the browser numerical comparison (about 8.13% relative L2 difference versus its CPU reference). The larger FP32 graph passed. We did not relax that failed check or claim the smaller graph was verified.

Other repaired engineering failures: CPU `accuracy_level=4` permitted different arithmetic from the browser; level 0 produced matched layer outputs. Casting all BF16 embeddings to FP16 was not exact, so embeddings are stored as original BF16 bits. Initial 128/256-token text runs reached their budgets; the 384-token run stopped normally at token 271. Missing package dependencies, browser module imports and the local tokenizer loader were fixed. GPU access for the original-model control required leaving the filesystem sandbox; no credentials or paid inference were used.

## What this does and does not establish

We now have a concrete exporter and a complete text-model browser execution proof. We no longer need to describe the problem merely as “no exporter available.”

The weights are the pinned upstream BF16 model plus the translated LoRA update, newly quantized for this runtime. They are **not bit-identical to the original MLX 4-bit base**. This work measures conversion/runtime behavior on synthetic fixtures, not JEE accuracy or preservation across a representative suite.

The public site still runs the separate 0.8B demo. Converted 9B weights have not been uploaded. Only one adapter has been exercised. Arbitrary browser image preprocessing, dynamic image sizes, all historical checkpoint choices, mobile memory/performance and a reproducible public 9B package remain unfinished. The immediate fidelity check is to isolate the original image failure and the new output change using matched image features, intermediate states and a precision/quantization control.

## Code and evidence

Prototype source and a sanitized summary are in [conversion_lab/browser_proto](../conversion_lab/browser_proto/README.md). Full local graphs and process logs are retained in the project recovery record; they are not included in GitHub. No private JEE questions, evaluation answers, user attempts, optimizer states or unrelated project files were published.
