# Original MLX quantization conversion — reference source

These are the exact source snapshots used for the preserved-weight export and native controls, retained for inspection. They are engineering scripts from the project workspace, **not a turnkey command for a fresh public clone**. They expect its frozen source-download metadata, synthetic diagnostic fixtures and original adapter layout; none of the private JEE question sets is needed or published here.

Dependencies include the helper modules in `../browser_proto/` and the synthetic-fixture controls in `../image_diagnostics/`. The original project layout sets the root two levels above the scripts. Preserve source identities and choose a new experiment directory when adapting these paths; the exporter intentionally refuses to resume with changed source identities.

The browser-ready model package is [hosted separately](https://huggingface.co/OmTheLast/LlamaJEE-9B-ONNX-WebGPU). Its manifest records original model revision, adapter source hashes, every binary hash and the explicit graph input shapes. The working browser implementation is in [`../../web/`](../../web/). See the [release report](../../research/EXPORT09B04_ORIGINAL_BROWSER_RELEASE.md) for completed checks and limits.

No optimizer updates or newly trained weights were produced by this conversion. The earlier requantized prototype remains preserved alongside this source.
