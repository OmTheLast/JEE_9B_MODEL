# Public browser deployment status — 23 October 2026

The requested destination is https://jee.ompatnaik.com/. The site is static: the visitor downloads model files and computes answers on their own device. There is no inference API, API key, localhost dependency, or question-upload endpoint. Hosting and model-file downloads still use the internet.

## What can run today

The existing preview uses unchanged `onnx-community/Qwen3.5-0.8B-ONNX`, pinned at `c0d619322dad7c4441a8841a53fc59772ddddcc0`, through Transformers.js 4.3.0. It is a separate small demonstration, not one of our trained 9B checkpoints. The website now displays this distinction prominently.

## What remains unfinished for trained 9B

Our MLX adapters are preserved in the public archive. MLX-to-PEFT mapping and small native text/image parity checks previously passed; see [the export report](EXPORT09B01_STAGE_B_RESULT.md). This does not establish ONNX or browser parity.

Fresh public-source checks on 23 October:

- [Transformers.js exporter request #1574](https://github.com/huggingface/transformers.js/issues/1574) remains open; its two comments provide no conversion implementation.
- The inspected current [Optimum ONNX model configurations](https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/model_configs.py) contain no `qwen3_5` registration.
- [onnx-community/Qwen3.5-9B-Onnx](https://huggingface.co/onnx-community/Qwen3.5-9B-Onnx/tree/40112c9ca55989bc0405ddf7671a6c15a99a8924) is packaged for `onnxruntime-genai`, with `genai_config.json`, `model.onnx` and external weights. It is not a drop-in multimodal Transformers.js package and does not contain our adapters.
- The Hugging Face model API search for Qwen3.5-9B tagged `transformers.js` returned no candidates. This is a bounded discovery result, not proof that conversion is impossible.

Changing the domain does not resolve this model-format problem. We have not exported, uploaded or browser-tested trained 9B weights in this round. There is no new model accuracy result.

## Required next engineering work

1. Establish a reproducible browser graph for the exact 9B architecture, including its recurrent state and vision processor. A custom exporter or carefully verified graph conversion is an engineering task, not a filename change.
2. Convert a named archived adapter, preserve its identity, and verify its update is active in the exported weights.
3. Compare native and browser results on fixed synthetic text/image fixtures; record numerical differences, completion and peak memory.
4. Publish the verified browser weights and add the model selector only after that path works. Measure device support; do not promise a 9B model fits phones merely because the page loads there.

Existing local harnesses remain as historical development tools, but are not the requested public runtime. A hosted inference service or a tunnel to a Mac would violate the browser-only requirement and is not used.
