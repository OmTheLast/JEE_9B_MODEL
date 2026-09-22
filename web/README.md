# JEE browser solver preview

Static browser site for the experimental JEE solver, built with Vite and Transformers.js 4.3.0. The browser downloads the pinned `onnx-community/Qwen3.5-0.8B-ONNX` model and runs inference with WebGPU. The model is an **unchanged 0.8B browser demonstrator**, not one of the trained 9B MLX LoRA adapters. Question text and images stay in the browser; Hugging Face serves model files. A one-time model download is roughly 850 MB.

The client-side harness builds a JEE-specific prompt with subject and thermodynamics convention hints, requests a decomposition plan/working/check/final answer, streams output, and checks only for the requested answer structure. It does not independently verify mathematical correctness or provide a calculator yet. It makes no benchmark or production-readiness claim.

```sh
cd jee/src/jee/web
npm ci
npm run check
npm run build
npm run dev
```

Open the local URL printed by Vite. A WebGPU-capable browser is required. Model loading starts only after clicking **Load model locally**. The static `dist/` folder can be hosted on GitHub Pages or Cloudflare Pages; use the built assets rather than source files. The currently bundled ONNX WASM asset is larger than Cloudflare Pages' 25 MiB single-file limit, so a Pages deployment needs a different WASM delivery path. The intended custom domain also requires DNS/hosting configuration.

Model: [ONNX Qwen3.5 0.8B](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX), pinned revision `c0d619322dad7c4441a8841a53fc59772ddddcc0`. Research archive: [JEE_9B_MODEL](https://github.com/OmTheLast/JEE_9B_MODEL). No JEE question corpus, gold answers, private review data or local user interactions are bundled in this website.
