"""Load one experimental JEE LoRA adapter with the pinned MLX Qwen base.

This script is copied into the public adapter archive as load_mlx.py. It is
deliberately explicit: these checkpoints are MLX-VLM LoRA tensors, not PEFT
or a standalone Hugging Face Transformers model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_vlm.generate import stream_generate
from mlx_vlm.trainer.lora_layers import LoRALinear
from mlx_vlm.utils import load


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_experiment(repo_root: Path, adapter_name: str):
    release = json.loads((repo_root / "release.json").read_text())
    entry = release["adapters"].get(adapter_name)
    if entry is None:
        raise ValueError(f"Unknown adapter: {adapter_name}")
    folder = repo_root / "adapters" / adapter_name
    settings = json.loads((folder / "jee_adapter.json").read_text())
    adapter_file = folder / "adapter.safetensors"
    if file_hash(adapter_file) != entry["sha256"]:
        raise ValueError("Adapter hash differs from the release manifest")
    base = release["base"]
    model_dir = snapshot_download(repo_id=base["repo"], revision=base["revision"])
    model, processor = load(model_dir, trust_remote_code=False)
    model.freeze()
    for layer in model.language_model.layers[-settings["last_language_layers"]:]:
        for name in settings["mlp_projections"]:
            wrapped = LoRALinear.from_base(
                getattr(layer.mlp, name), r=settings["rank"],
                scale=settings["scale"], dropout=settings["dropout"],
            )
            wrapped.linear.freeze()
            setattr(layer.mlp, name, wrapped)
    weights = mx.load(str(adapter_file))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError("Adapter tensors do not match the pinned base and LoRA scope")
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if args.image and not args.image.is_file():
        parser.error("--image must name an existing file")
    model, processor = load_experiment(Path(__file__).resolve().parent, args.adapter)
    content = ([{"type": "image"}] if args.image else []) + [{
        "type": "text",
        "text": ("Solve this question. Choose the decisive method, verify once, then stop. "
                 "If information is missing, say so.\n\n" + args.question),
    }]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    for part in stream_generate(
        model, processor, prompt, image=[str(args.image)] if args.image else [],
        max_tokens=args.max_tokens, temperature=0,
        enable_thinking=False, skip_special_tokens=True,
    ):
        print(part.text or "", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
