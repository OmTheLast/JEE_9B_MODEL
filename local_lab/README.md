# Original local 9B JEE lab

This is the original interactive harness, packaged for this public repository. It runs the pinned MLX Qwen3.5-9B base **on an Apple Silicon Mac** and can load three saved experimental LoRA adapters from `../adapters/`. The browser page is only the interface; Python and MLX run the 9B model locally. It is distinct from the `../web/` Transformers.js demo, which runs an unchanged 0.8B ONNX model inside the browser.

The interface accepts question text or a PNG/JPEG, streams raw model output, labels length stops, checks the syntax of a final answer and lets you reopen saved attempts. The syntax check **does not check scientific correctness**. This is an exploratory lab, not the final tool-backed decomposition/calculation/verification harness or an exam benchmark. None of the adapters was promoted as a better general JEE solver.

## Run it

From the repository root on a compatible Apple Silicon Mac:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r local_lab/requirements.txt
.venv/bin/python local_lab/start.py
```

Open <http://127.0.0.1:8766/>. The first model use downloads the pinned public `mlx-community/Qwen3.5-9B-4bit` snapshot from Hugging Face (about 5.5 GiB of weights before runtime overhead); keep enough disk space and memory available. The startup command checks whether the server is already listening and starts it in the background if needed. After a reboot, run it again. To see foreground errors, use `.venv/bin/python local_lab/server.py --port 8766` instead. Do not open `app.html` as a `file://` page; that bypasses the server.

Model choices are unchanged base, 30-family/last-three-layer step 60, 30-family/last-sixteen-layer step 60 and 60-family/last-three-layer step 60. A *family* is one independent source problem and its related training views; a *step* is one optimizer update. The base is the publicly post-trained 9B checkpoint without our adapters. The adapters are MLX-specific; the browser 0.8B demo cannot load them.

## Attempt history and privacy

This copy creates `local_lab/runs/<run-id>/` on your Mac. It saves a normalized request and optional image, model identity, streamed progress every 16 tokens, final result or halt record, and worker logs. The page lists the latest 30 attempts and reopens a selected attempt. `local_lab/runs/` is Git-ignored. **No historical user questions, screenshots, private evaluation keys, or local interaction logs are in this public repository.** The original development workspace keeps its own existing history separately. Do not submit protected benchmark questions to this exploratory interface.

The server binds to `127.0.0.1`, checks the local Host/Origin and allows one generation at a time. If generation is interrupted, the last saved progress is visible, but work since the last receipt may be lost. Model files come from Hugging Face; submitted question data stay with the local server.

## Source and checks

`server.py`, `worker.py`, `app.html` and `start.py` are adapted from the original lab source. Public packaging replaces private project paths with the repository's `release.json`, `adapters/` and pinned base download. The same hash and tensor-shape checks in `load_mlx.py` apply to selected adapters. The code requires no training dataset or optimizer states.

Run fast checks without loading 9B weights:

```sh
.venv/bin/python -m unittest discover -s local_lab -p 'test_*.py' -v
```

For a historical explanation of results and limits, read [EXPERIMENT_HISTORY.md](../EXPERIMENT_HISTORY.md) and the [charts](../charts/README.md).
