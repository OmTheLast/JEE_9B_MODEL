#!/usr/bin/env python3
"""Convert the project's MLX-VLM LoRA tensors to a standard PEFT adapter.

This converts only the low-rank delta. It does not download, fuse, quantize or
export the base model. The produced adapter must still pass a native-vs-PEFT
parity run against the exact upstream base before it is treated as portable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


KEY_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.lora_(?P<side>a|b)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def convert(
    source: Path,
    settings_path: Path,
    output_dir: Path,
    base_model: str,
    base_revision: str,
) -> dict:
    settings = json.loads(settings_path.read_text())
    rank = int(settings["rank"])
    scale = float(settings["scale"])
    alpha_float = scale * rank
    if not alpha_float.is_integer():
        raise ValueError("PEFT lora_alpha would not be integral; add explicit compatibility handling")
    alpha = int(alpha_float)
    source_tensors = load_file(source)
    converted: dict[str, np.ndarray] = {}
    layers: set[int] = set()
    projections: set[str] = set()
    pairs: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    mapping: list[dict] = []

    for old_key, tensor in sorted(source_tensors.items()):
        match = KEY_RE.fullmatch(old_key)
        if not match:
            raise ValueError(f"Unexpected MLX adapter key: {old_key}")
        layer = int(match.group("layer"))
        projection = match.group("projection")
        side = match.group("side")
        peft_side = side.upper()
        new_key = (
            "base_model.model.model.language_model.layers."
            f"{layer}.mlp.{projection}.lora_{peft_side}.weight"
        )
        value = np.ascontiguousarray(tensor.T)
        converted[new_key] = value
        layers.add(layer)
        projections.add(projection)
        pairs.setdefault((layer, projection), {})[side] = tensor
        mapping.append(
            {
                "mlx_key": old_key,
                "mlx_shape": list(tensor.shape),
                "peft_key": new_key,
                "peft_shape": list(value.shape),
                "operation": "transpose",
            }
        )

    expected_layers = list(range(32 - int(settings["last_language_layers"]), 32))
    if sorted(layers) != expected_layers:
        raise ValueError(f"Adapter layers {sorted(layers)} do not match settings {expected_layers}")
    if projections != set(settings["mlp_projections"]):
        raise ValueError("Adapter projection names do not match settings")
    if len(converted) != len(layers) * len(projections) * 2:
        raise ValueError("Incomplete LoRA A/B tensor pairs")

    rng = np.random.default_rng(20260923)
    max_abs_error = 0.0
    max_rel_error = 0.0
    parity_rows = []
    for (layer, projection), pair in sorted(pairs.items()):
        if set(pair) != {"a", "b"}:
            raise ValueError(f"Missing A/B tensor for layer {layer} {projection}")
        mlx_a = pair["a"].astype(np.float32)
        mlx_b = pair["b"].astype(np.float32)
        peft_a = mlx_a.T
        peft_b = mlx_b.T
        x = rng.standard_normal((2, mlx_a.shape[0]), dtype=np.float32)
        mlx_result = scale * ((x @ mlx_a) @ mlx_b)
        peft_result = (scale * (x @ peft_a.T)) @ peft_b.T
        absolute = float(np.max(np.abs(mlx_result - peft_result)))
        denominator = max(float(np.max(np.abs(mlx_result))), 1e-12)
        relative = absolute / denominator
        max_abs_error = max(max_abs_error, absolute)
        max_rel_error = max(max_rel_error, relative)
        parity_rows.append(
            {
                "layer": layer,
                "projection": projection,
                "max_abs_error": absolute,
                "max_relative_error": relative,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    weights_path = output_dir / "adapter_model.safetensors"
    save_file(converted, weights_path, metadata={"format": "pt"})
    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": "layers",
        "layers_to_transform": sorted(layers),
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_bias": False,
        "lora_dropout": float(settings["dropout"]),
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.21.0",
        "qalora_group_size": 16,
        "r": rank,
        "rank_pattern": {},
        "revision": base_revision,
        "target_modules": sorted(projections),
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    atomic_json(output_dir / "adapter_config.json", config)
    manifest = {
        "status": "representation_converted_not_end_to_end_verified",
        "source": str(source),
        "source_sha256": sha256(source),
        "settings": str(settings_path),
        "settings_sha256": sha256(settings_path),
        "base_model": base_model,
        "base_revision": base_revision,
        "rank": rank,
        "mlx_scale": scale,
        "peft_lora_alpha": alpha,
        "layers": sorted(layers),
        "projections": sorted(projections),
        "tensor_count": len(converted),
        "mapping": mapping,
        "randomized_delta_parity": {
            "seed": 20260923,
            "rows": parity_rows,
            "max_abs_error": max_abs_error,
            "max_relative_error": max_rel_error,
        },
        "adapter_model_sha256": sha256(weights_path),
        "limitations": [
            "No upstream base weights were loaded.",
            "No PEFT forward pass, fused-model generation, ONNX export or browser run was performed.",
            "The MLX adapter was trained against a 4-bit base; applying its delta to BF16 may change behavior.",
        ],
    }
    atomic_json(output_dir / "conversion_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()
    if not args.source.is_file() or not args.settings.is_file():
        parser.error("--source and --settings must name files")
    if args.output.exists():
        parser.error("--output must not already exist")
    result = convert(args.source, args.settings, args.output, args.base_model, args.base_revision)
    print(json.dumps({k: result[k] for k in ("status", "tensor_count", "adapter_model_sha256")}, indent=2))


if __name__ == "__main__":
    main()
