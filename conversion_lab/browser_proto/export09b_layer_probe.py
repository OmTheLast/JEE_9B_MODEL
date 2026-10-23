"""Bounded real-weight Qwen3.5-9B decoder export probe, not a complete solver.

One-token graphs make recurrent state explicit. Tests compare the custom graph
against the installed Transformers implementation, then ONNX Runtime. Inputs are
synthetic hidden states; no JEE evaluation or training data is consumed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
from checkpoint import atomic_write

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/jee/training/next01/export09b01'
OUT = ROOT / 'data/jee/training/next01/export09b02'


class RecurrentLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, conv, state):
        layer = self.layer
        a = layer.linear_attn
        h = layer.input_layernorm(x)
        qkv = a.in_proj_qkv(h).transpose(1, 2)
        conv = torch.cat((conv[:, :, 1:], qkv), dim=-1)
        qkv = torch.nn.functional.silu((conv * a.conv1d.weight.squeeze(1)).sum(-1))
        q, k, v = torch.split(qkv, [a.key_dim, a.key_dim, a.value_dim], dim=-1)
        q = q.reshape(1, a.num_k_heads, a.head_k_dim).repeat_interleave(a.num_v_heads // a.num_k_heads, 1)
        k = k.reshape(1, a.num_k_heads, a.head_k_dim).repeat_interleave(a.num_v_heads // a.num_k_heads, 1)
        v = v.reshape(1, a.num_v_heads, a.head_v_dim)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / a.head_k_dim ** .5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = a.in_proj_b(h).sigmoid().reshape(1, a.num_v_heads, 1)
        decay = (-a.A_log.exp() * torch.nn.functional.softplus(a.in_proj_a(h) + a.dt_bias)).exp()
        state = state * decay.reshape(1, a.num_v_heads, 1, 1)
        delta = (v - (state * k.unsqueeze(-1)).sum(-2)) * beta
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        y = (state * q.unsqueeze(-1)).sum(-2)
        z = a.in_proj_z(h).reshape(-1, a.head_v_dim)
        y = a.norm(y.reshape(-1, a.head_v_dim), z).reshape(1, 1, a.value_dim)
        x = x + a.out_proj(y)
        return x + layer.mlp(layer.post_attention_layernorm(x)), conv, state


class AttentionLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, key, value, cos, sin):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
        layer = self.layer
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q, gate = a.q_proj(h).reshape(1, 1, -1, a.head_dim * 2).chunk(2, dim=-1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).reshape(1, 1, -1, a.head_dim)).transpose(1, 2)
        v = a.v_proj(h).reshape(1, 1, -1, a.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        key = torch.cat((key, k), dim=2)
        value = torch.cat((value, v), dim=2)
        scores = q @ repeat_kv(key, a.num_key_value_groups).transpose(2, 3) * a.scaling
        y = scores.softmax(-1) @ repeat_kv(value, a.num_key_value_groups)
        y = y.transpose(1, 2).reshape(1, 1, -1) * gate.reshape(1, 1, -1).sigmoid()
        x = x + a.o_proj(y)
        return x + layer.mlp(layer.post_attention_layernorm(x)), key, value


class ReferenceCache:
    def __init__(self, idx, inputs, recurrent):
        self.idx, self.recurrent = idx, recurrent
        if recurrent:
            self.layers = {idx: SimpleNamespace(record_past=False, conv_states=[inputs[1].clone()], recurrent_states=[inputs[2].clone()])}
        else:
            self.key, self.value = inputs[1].clone(), inputs[2].clone()

    def has_previous_state(self, *args, **kwargs):
        return True

    def update_recurrent_state(self, state, idx):
        self.layers[idx].recurrent_states[0] = state

    def update(self, key, value, idx):
        self.key = torch.cat((self.key, key), 2)
        self.value = torch.cat((self.value, value), 2)
        return self.key, self.value


def error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return {'max_abs': float(np.max(np.abs(a-b))), 'relative_l2': float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, choices=[29, 31], required=True)
    parser.add_argument('--version', default='v1', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--accuracy-level', type=int, choices=[0, 4], default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(709)
    logging.getLogger().setLevel(logging.WARNING)
    out = OUT / f'layer{args.layer:02d}-{args.version}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise RuntimeError('Immutable probe already exists; inspect before creating a new version')
    result = {'status': 'started', 'layer': args.layer, 'started': time.time(), 'scope': 'One actual trained decoder layer; not a full model or accuracy test'}
    def save(status):
        result['status'] = status
        atomic_write(out / 'progress.json', json.dumps(result, indent=2)+'\n')
    save('loading-real-weights')
    config = AutoConfig.from_pretrained(SOURCE / 'upstream-bf16', local_files_only=True).text_config
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        layer = Qwen3_5DecoderLayer(config, args.layer)
    index = json.loads((SOURCE / 'upstream-bf16/model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.language_model.layers.{args.layer}.'
    tensors = {}
    for key, shard in index.items():
        if key.startswith(prefix):
            with safe_open(SOURCE / 'upstream-bf16' / shard, framework='pt') as f:
                tensors[key[len(prefix):]] = f.get_tensor(key).float()
    layer.load_state_dict(tensors, assign=True, strict=True)
    layer.eval()
    recurrent = config.layer_types[args.layer] == 'linear_attention'
    model = RecurrentLayer(layer) if recurrent else AttentionLayer(layer)
    model.eval()
    def inputs(past, random_state):
        x = torch.randn(1, 1, config.hidden_size) * .3
        if recurrent:
            a = layer.linear_attn
            return (x, torch.randn(1, a.conv_dim, a.conv_kernel_size)*random_state,
                    torch.randn(1,a.num_v_heads,a.head_k_dim,a.head_v_dim)*random_state)
        rotary_dim = int(config.head_dim * config.rope_parameters['partial_rotary_factor'])
        angles = torch.randn(1,1,rotary_dim//2)
        angles = torch.cat((angles,angles),-1)
        return (x, torch.randn(1,config.num_key_value_heads,past,config.head_dim)*random_state,
                torch.randn(1,config.num_key_value_heads,past,config.head_dim)*random_state, angles.cos(), angles.sin())
    samples = [inputs(0,0), inputs(3,.01), inputs(9,.02)]
    base_output = model(*samples[1])[0].numpy().copy()
    adapter_file = SOURCE / 'peft-reliable01-step060-v1/adapter_model.safetensors'
    result['adapter_sha256'] = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    with safe_open(adapter_file, framework='pt') as f:
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            p = f'base_model.model.{prefix}mlp.{name}.'
            delta = f.get_tensor(p+'lora_B.weight').float() @ f.get_tensor(p+'lora_A.weight').float() * 2
            getattr(layer.mlp, name).weight.add_(delta)
    result['adapter_effect'] = error(model(*samples[1])[0].numpy(), base_output)
    assert result['adapter_effect']['max_abs'] > 0
    result['reference_checks'] = []
    for sample in samples:
        cache = ReferenceCache(args.layer, sample, recurrent)
        ref = layer(sample[0], position_embeddings=None if recurrent else (sample[3], sample[4]), past_key_values=cache)
        wanted = (ref, cache.layers[args.layer].conv_states[0], cache.layers[args.layer].recurrent_states[0]) if recurrent else (ref, cache.key, cache.value)
        actual = model(*sample)
        checks = [error(a.numpy(), b.numpy()) for a,b in zip(actual,wanted)]
        assert all(np.allclose(a.numpy(),b.numpy(),rtol=2e-4,atol=2e-5) for a,b in zip(actual,wanted)), checks
        result['reference_checks'].append(checks)
    save('reference-parity-passed')
    names = ['hidden','conv','state'] if recurrent else ['hidden','key','value','cos','sin']
    outputs = ['hidden_out','conv_out','state_out'] if recurrent else ['hidden_out','key_out','value_out']
    dynamic = None if recurrent else {n:{2:'past_length'} for n in ['key','value']}
    if dynamic: dynamic.update({n:{2:'present_length'} for n in ['key_out','value_out']})
    path = out / 'layer.fp32.onnx'
    torch.onnx.export(model, samples[1], str(path), input_names=names, output_names=outputs,
                      opset_version=18, dynamo=False, dynamic_axes=dynamic, do_constant_folding=True)
    expected = [[v.numpy().copy() for v in model(*s)] for s in samples]
    feeds = [{n:t.numpy().copy() for n,t in zip(names,s)} for s in samples]
    del model, layer, tensors, delta
    gc.collect()
    save('onnx-exported')
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    result['onnx_checks'] = []
    for feed, target in zip(feeds, expected):
        values = session.run(None,feed)
        result['onnx_checks'].append([error(a,b) for a,b in zip(values,target)])
        assert all(np.allclose(a,b,rtol=3e-4,atol=3e-5) for a,b in zip(values,target))
    del session; gc.collect()
    save('onnx-native-parity-passed')
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    result['accuracy_level'] = args.accuracy_level
    quantizer = MatMulNBitsQuantizer(str(path), block_size=32, is_symmetric=True, accuracy_level=args.accuracy_level)
    quantizer.process()
    qpath = out / 'layer.q4.onnx'
    quantizer.model.save_model_to_file(str(qpath), use_external_data_format=False)
    del quantizer; gc.collect()
    session = ort.InferenceSession(str(qpath), options, providers=['CPUExecutionProvider'])
    qvalues = session.run(None,feeds[1])
    result['q4_vs_fp32'] = [error(a,b) for a,b in zip(qvalues,expected[1])]
    # Browser parity compares the SAME quantized model; q4 fidelity is separate.
    fixture = {'inputs':{},'outputs':{}}
    for group, values in [('inputs',feeds[1]), ('outputs',dict(zip(outputs,qvalues)))]:
        for name,value in values.items():
            filename = group+'-'+name+'.f32'
            value.astype('<f4').tofile(out/filename)
            fixture[group][name] = {'file':filename,'dims':list(value.shape),'dtype':'float32'}
    (out/'browser-fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    result['files'] = {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [path,qpath]}
    result['elapsed_seconds'] = time.time()-result['started']
    save('complete-native-browser-test-pending')
    atomic_write(out/'result.json',json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    main()
