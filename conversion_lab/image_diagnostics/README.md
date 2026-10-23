# Image-conversion diagnostic source snapshots

See [the result](../../research/EXPORT09B03_IMAGE_DIAGNOSTICS.md). These are engineering scripts from the project workspace, not an installable solver or a completed Transformers.js model.

The scripts expect the original project layout under `src/jee`, the pinned upstream and MLX files recorded in earlier conversion reports, prepared private synthetic controls, and the isolated Python dependencies from the [browser prototype](../browser_proto/). Shared helpers (`checkpoint.py`, `export09b_layer_probe.py`, and MLX parity helpers) remain in the prior conversion sources. The fixture generator uses a macOS Arial font.

Run fixtures before vision, original references before decoder checks, and use the documented memory/time guards. Do not treat a partial file or failed process as a completed measurement. The HTML requires a local static server, original-weight projection, diagnostic arrays and ORT WebGPU runtime; it is a single-projection test, not the public model selector.

All questions in this diagnostic were generated synthetically. No private JEE questions, raw hidden-state arrays, credentials or model weights are published here. The public site still uses its separate 0.8B preview.
