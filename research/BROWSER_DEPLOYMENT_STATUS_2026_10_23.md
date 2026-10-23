# Public browser deployment status — 23 October 2026

**Path correction, 23 October 2026:** The user clarified that the canonical destination is **https://ompatnaik.com/llamajee/**, a subdirectory of the main website. The subdomain entries below are historical. The static build is copied into the portfolio repository’s `llamajee/` directory; its root CNAME and other pages are preserved. Model availability is unchanged.

**Domain correction, 23 October 2026:** The project address is now https://llamajee.ompatnaik.com/, matching the LlamaJEE.js repository name. Cloudflare DNS and GitHub Pages have been updated. References to the former `jee` hostname below describe the earlier deployment checks; they are preserved as history. The new hostname has an approved certificate, HTTPS enforcement is enabled, and the page returned HTTP 200 over verified HTTPS and rendered in Chrome. No new model generation was tested; trained 9B browser execution remains incomplete.

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

## Domain and page verification

The site is now branded **LlamaJEE.js**, matching this repository. A notebook-style home page, solver workspace, model availability cards and preserved experiment graphs are published. The project name does not mean the underlying model is Meta Llama; the current checkpoints use Qwen.

GitHub Pages serves the static build from `main/docs`. The custom hostname is configured, and public DNS confirms `jee.ompatnaik.com` is a CNAME to `omthelast.github.io`. The public HTTP response contains the new title, notebook and explicit 9B-pending label. The build and all four existing harness tests pass. The unchanged model worker bundle retains its previous hash; no inference route was added.

At the last check in this round, GitHub was still provisioning the domain's HTTPS certificate after an initial cached DNS-not-found response. A single documented hostname revalidation was performed after public DNS propagated. HTTPS has not yet been verified, so the solver is not declared ready on this hostname: WebGPU requires a secure context. Do not bypass certificate warnings or turn off TLS verification. The next hosting check is to verify issuance and enable HTTPS enforcement, followed by a public-browser smoke test.

Desktop layout was inspected in Chrome and one positioning defect was corrected. Responsive CSS is included; the browser automation's attempted viewport override did not change the measured desktop width, so mobile rendering is not claimed as verified. No new model inference or JEE accuracy experiment ran in this round.
