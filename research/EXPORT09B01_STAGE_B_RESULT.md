# 9B portable-forward result: adapter conversion works; browser export is blocked

23 September 2026. This experiment used the rejected/unselected Reliable01 step60 adapter only as a format fixture. It does not promote that checkpoint or measure JEE improvement.

## Result in plain English

We successfully translated the small learned LoRA update from the project MLX format into standard Hugging Face PEFT form. The translated adapter runs on the original upstream Qwen3.5-9B BF16 model and behaves consistently with the MLX version on one synthetic text problem and one synthetic image problem.

The next link is blocked: the current public toolchains can **run** community Qwen3.5 ONNX models in Transformers.js, but they do not publish a reproducible Qwen3.5 exporter for our newly fused weights. Therefore we did not create a19GB fused copy or claim a browser model. This is the useful result of trying the9B path first.

## Verified chain

| Gate | Result |
|---|---|
| MLX LoRA → PEFT tensor mapping | Pass:18/18 tensors; exact randomized low-rank delta equality |
| Pinned upstream download | Pass:13 required files; byte sizes and LFS/Git object hashes match revision `c202236…` |
| Prompt string and token IDs across MLX/Transformers | Pass on text and image fixtures |
| Converted adapter loads under PEFT | Pass |
| Base top-one next token across runtimes | Pass on2/2 fixtures |
| Adapted top-one next token across runtimes | Pass on2/2 fixtures |
| Top-five agreement | Four shared token IDs on each condition/fixture; expected logit differences from4-bit versus BF16 |
| Generated result | Base/adapted × MLX/BF16 all stop normally with `FINAL_ANSWER: 9` |
| Browser-compatible ONNX exporter | Blocked in current inspected public toolchains |

The MLX runs peaked at about6.97GiB reported allocation and took2.6–3.6 seconds per generation after load. Portable BF16 MPS generations took10.9–11.1 seconds and reached20,032,274,432 driver-allocated bytes, about18.66GiB. These are synthetic engineering timings, not phone/browser claims.

## Runtime problem found and fixed

Transformers5.17's default asynchronous loader dispatched four huge BF16 tensors to MPS together. It advanced only3/760 parameters in about five minutes while four threads spun inside Metal copy-kernel setup. After setting `HF_DEACTIVATE_ASYNC_LOAD=1`, all760 parameters loaded in about3 seconds and the full parity script completed. The runner now sets this explicitly. This was a loading-path defect, not an out-of-memory event; memory and swap stayed stable.

The first Hugging Face download command also combined `--include` with positional patterns. The CLI warned that the include filter was ignored, downloaded the large weights and skipped JSON processor/config files. Fetching `--include '*.json'` separately completed the package without redownloading the weights. The final13-file verification passed.

## Why ONNX stops here

Pinned source inspection found:

- Optimum-ONNX commit `ebfc7c3…` registers `qwen3` but no `qwen3_5` ONNX configuration and contains no Qwen3.5 Python exporter code.
- Transformers.js commit `836b9cb…` contains Qwen3.5 runtime classes and tests, but no conversion/Python script under `scripts/`; its public issue requesting the Qwen3.5 exporter remains open.
- The available community9B artifact is an ONNX Runtime GenAI graph, not the Transformers.js multimodal packaging needed by the website.

This proves a current reproducibility blocker, not theoretical impossibility. The exact inspected repository commits are recorded above; local raw process and model receipts are excluded from this public repository.

## Decision for the4B program

Do not assume Qwen3.5-4B wins deployment merely because an unchanged community ONNX repository exists. Its baseline can enter the browser smoke, but before training selection it must demonstrate one of:

1. a released/reproducible Qwen3.5 exporter;
2. verified injection of our fused weights into a matching full-precision Transformers.js ONNX graph followed by quantization; or
3. a repeat of the selected training recipe on an architecture with a supported portable exporter.

Run model selection next. The deployment gate now has real evidence, and the adapter/data work remains valid regardless of which4B-class model wins.

## Public artifact and excluded raw receipts

- [`conversion_lab/convert_mlx_lora_to_peft.py`](../conversion_lab/convert_mlx_lora_to_peft.py) is the published representation converter.
- The exact candidate identities/revisions are in [`candidate-manifest-v1.json`](candidate-manifest-v1.json).
- Local raw receipts, downloaded19GB upstream weights, generated synthetic fixture, process samples and the unselected converted adapter are deliberately excluded from GitHub. Their verified measurements and hashes are summarized above; the adapter remains in the existing experimental Hugging Face history archive in native MLX form.

No optimizer update, protected evaluation access, fused-model save, ONNX file, browser claim or remote model upload occurred.
