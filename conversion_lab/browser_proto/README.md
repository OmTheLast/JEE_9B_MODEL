# Custom 9B ONNX browser prototype

Read [the results and failed gates](../../research/EXPORT09B02_BROWSER_PROTOTYPE.md) before using this research code. This is a development prototype, not a released model loader.

## Inputs and layout

Run from the repository root, in an isolated Python environment with the versions in `package-versions.json`. The scripts expect the pinned `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` under `data/jee/training/next01/export09b01/upstream-bf16/`, and the previously converted Reliable01 step-60 PEFT files under `peft-reliable01-step060-v1/` alongside it. See the preceding export report and representation converter. These large files are not included here.

1. `export09b_layer_probe.py --layer 29 --version v3 --accuracy-level 0` and the layer31 probe validate real-weight component graphs against Transformers and native ONNX Runtime.
2. `export09b_text_proto.py` creates independently recorded shards in `data/jee/training/next01/export09b02/text-q4-v2/`. Export resumes only when source/config/adapter identities and artifact hashes match. Source changes require a new version.
3. `run_export09b_text_proto.py --max-new-tokens 384` runs the text CPU control. `export09b_vision_probe.py` creates the fixed-resolution image encoder. `make_export09b_image_fixture.py` prepares the synthetic image control; the runner’s `--image --max-new-tokens 64` option exercises it.
4. The HTML prototypes expect a static test root containing those exported folders plus a `runtime/` directory with the recorded ONNX Runtime Web distribution, a `transformers/` distribution and the pinned tokenizer JSON/config files under `tokenizer/`. They compute in the browser. Serve only that test folder, not the private project root. The text page accepts `?image=1` for the fixed preprocessed image fixture.

The image fixture is preprocessed in Python; arbitrary user screenshots are not implemented. Current scripts specialize to this model’s dimensions. Caller-side process deadlines were used for export/native checks; the browser has a ten-minute deadline and a stop control. Browser peak memory remains unmeasured. Do not put these large test artifacts into GitHub or treat a successful run as a JEE benchmark.

The inference code follows the installed Hugging Face Transformers Qwen3.5 model implementation and uses ONNX Runtime’s quantizer. Upstream model licensing and dependency licenses still apply. The exact input identities and bounded measurements are in `summary.json`; it contains no private evaluation material.
