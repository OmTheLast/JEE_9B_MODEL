"""One local exploratory JEE inference attempt; streams events and saves progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.generate import stream_generate
from mlx_vlm.utils import load
from huggingface_hub import snapshot_download

from storage import atomic_write
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_mlx import load_experiment


ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / "release.json").read_text())
FINAL_RE = re.compile(r"(?m)^FINAL_ANSWER:\s*(\{[^\n]*\})\s*$")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MODEL_CHOICES = {
    "base": None,
    "reliable01-step60": "reliable01-step060",
    "coverage16-step60": "coverage16-step060",
    "data60-step60": "data60-step060",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def parse_final(text: str, request: dict | None = None) -> dict:
    matches = FINAL_RE.findall(text)
    if len(matches) != 1:
        return {"status": "missing_or_multiple_final_lines", "count": len(matches)}
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return {"status": "invalid_json"}
    if not isinstance(value, dict) or value.get("type") not in ("options", "numeric", "abstain"):
        return {"status": "invalid_shape"}
    if value["type"] == "options":
        selected = value.get("selected_options")
        if not (isinstance(selected, list) and selected and
                all(isinstance(x, str) and x for x in selected) and len(set(selected)) == len(selected)):
            return {"status": "invalid_options"}
    if value["type"] == "numeric":
        number = value.get("value")
        if not isinstance(number, str) or not re.fullmatch(r"-?\d+(?:\.\d+)?", number):
            return {"status": "invalid_numeric_format"}
        if request and request.get("exam") == "main" and not re.fullmatch(r"-?\d+", number):
            return {"status": "main_numeric_not_integer"}
    if request and request.get("answer_type") in ("numeric", "options") and value["type"] not in (
        request["answer_type"], "abstain"
    ):
        return {"status": "answer_type_mismatch", "value": value}
    return {"status": "parseable", "value": value}


def prompt_for(processor, request: dict, has_image: bool) -> str:
    instructions = (
        "Solve the supplied JEE problem efficiently. State only assumptions or sign "
        "conventions that change the answer. Choose the decisive method, carry out its "
        "necessary steps, check the key result once, then stop. Do not restate the "
        "question, repeat settled calculations, or explore alternatives without a reason. "
        "For a multiple-correct question, check every option. If the question could "
        "genuinely belong to physics or chemistry and their conventions change the "
        "interpretation, explain both. Do not invent missing information. End with exactly "
        "one line FINAL_ANSWER: followed by a JSON object. "
        'For options use {"type":"options","selected_options":["label"]}; for a numeric '
        'answer use {"type":"numeric","value":"decimal string"}; if it cannot be '
        'answered use {"type":"abstain"}. Use the option labels shown in the question.'
    )
    visible = {
        "subject_hint": request["subject"],
        "exam_hint": request["exam"],
        "answer_type_hint": request["answer_type"],
        "question": request["question"],
    }
    content = ([{"type": "image"}] if has_image else []) + [
        {"type": "text", "text": instructions + "\nProblem: " + json.dumps(visible, ensure_ascii=False)}
    ]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def checked_model(choice: str):
    base = RELEASE["base"]
    adapter_name = MODEL_CHOICES[choice]
    if adapter_name is None:
        model_dir = snapshot_download(repo_id=base["repo"], revision=base["revision"])
        model, processor = load(model_dir, trust_remote_code=False)
        model.freeze()
        model.eval()
        return model, processor, base["revision"], None
    model, processor = load_experiment(ROOT, adapter_name)
    return model, processor, base["revision"], RELEASE["adapters"][adapter_name]["sha256"]


def timeout(*_):
    raise TimeoutError("The 240-second generation deadline was reached")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    request = json.loads((run / "request.json").read_text())
    choice = request["model"]
    if choice not in MODEL_CHOICES:
        raise ValueError("Unknown model choice")
    started = time.monotonic()
    text = ""
    last = None
    try:
        mx.set_memory_limit(40 * 2**30)
        mx.set_cache_limit(2 * 2**30)
        mx.random.seed(0)
        model, processor, revision, adapter_sha = checked_model(choice)
        save(run / "model-identity.json", {
            "choice": choice, "revision": revision, "adapter_sha256": adapter_sha,
            "worker_sha256": sha(Path(__file__)),
        })
        image_path = run / "question.png"
        images = [str(image_path)] if image_path.exists() else []
        prompt = prompt_for(processor, request, bool(images))
        emit({"type": "ready", "model": choice, "run_id": run.name})
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(240)
        try:
            for part in stream_generate(
                model, processor, prompt, image=images,
                max_tokens=request["max_tokens"], temperature=0,
                enable_thinking=False, skip_special_tokens=True,
            ):
                fragment = part.text or ""
                text += fragment
                last = part
                if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ValueError("Response byte cap exceeded")
                if fragment:
                    emit({"type": "delta", "text": fragment})
                if part.generation_tokens and part.generation_tokens % 16 == 0:
                    save(run / "inprogress.json", {
                        "status": "running", "text": text,
                        "generation_tokens": part.generation_tokens,
                        "elapsed_seconds": time.monotonic() - started,
                    })
        except TimeoutError:
            status = "timeout"
        else:
            status = "completed"
        finally:
            signal.alarm(0)
        result = {
            "status": status,
            "model": choice,
            "text": text,
            "final_answer": parse_final(text, request),
            "generation_tokens": last.generation_tokens if last else None,
            "prompt_tokens": last.prompt_tokens if last else None,
            "finish_reason": last.finish_reason if last else None,
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_gb": last.peak_memory if last else None,
            "model_revision": revision,
            "adapter_sha256": adapter_sha,
            "question_sha256": sha(run / "request.json"),
            "image_sha256": sha(image_path) if images else None,
            "limits": ["Exploratory interaction only; final-answer parsing does not establish correctness."],
        }
        save(run / "result.json", result)
        (run / "inprogress.json").unlink(missing_ok=True)
        emit({"type": "done", "result": {k: result[k] for k in (
            "status", "final_answer", "generation_tokens", "finish_reason", "elapsed_seconds"
        )}})
    except BaseException as error:
        save(run / "HALTED.json", {
            "status": "halted", "error_type": type(error).__name__,
            "error": str(error), "partial_text": text,
            "elapsed_seconds": time.monotonic() - started,
        })
        emit({"type": "error", "message": str(error), "partial_saved": True})
        raise


if __name__ == "__main__":
    main()
