# MLX adapter portability lab

This folder contains the representation converter used in the documented 9B feasibility experiment. It converts this project's Qwen3.5 MLX-VLM LoRA A/B tensors to standard PEFT tensor names, orientations and scaling. It does **not** convert the base model, produce ONNX, or make an adapter browser-compatible by itself.

The tested Reliable01 fixture used rank4 and MLX scale2, which maps to PEFT `lora_alpha=8`. The script validates layer/projection coverage and runs randomized low-rank delta equality before writing a new output directory.

```sh
python convert_mlx_lora_to_peft.py \
  --source /path/to/adapter.safetensors \
  --settings /path/to/jee_adapter.json \
  --output /new/path/peft-adapter \
  --base-model Qwen/Qwen3.5-9B \
  --base-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

Requirements: Python, NumPy and `safetensors`. The output still requires full native PEFT parity against the exact upstream model before fusion. Read the [experiment result](../research/EXPORT09B01_STAGE_B_RESULT.md) for what passed and why browser export stopped.
