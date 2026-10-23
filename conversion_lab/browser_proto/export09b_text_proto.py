"""Experimental sharded text-only export of the pinned 9B + Reliable01 step60.

Uses the layer graph verified by export09b_layer_probe.py. This is a custom ORT
runner package, NOT a drop-in Transformers.js model. No vision export here.
"""
import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from export09b_layer_probe import SOURCE, OUT, RecurrentLayer, AttentionLayer
from export09b_fixtures import INSTRUCTION, QUESTION, sha256
from checkpoint import atomic_write


class Head(torch.nn.Module):
    def __init__(self, norm, weight):
        super().__init__()
        self.norm = norm
        self.weight = torch.nn.Parameter(weight)

    def forward(self, hidden):
        return torch.nn.functional.linear(self.norm(hidden), self.weight)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    logging.getLogger().setLevel(logging.ERROR)
    out = OUT / 'text-q4-v2'
    out.mkdir(exist_ok=True)
    config = AutoConfig.from_pretrained(SOURCE/'upstream-bf16', local_files_only=True).text_config
    config._attn_implementation = 'eager'
    index = json.loads((SOURCE/'upstream-bf16/model.safetensors.index.json').read_text())['weight_map']
    adapter_file = SOURCE/'peft-reliable01-step060-v1/adapter_model.safetensors'
    identity = {'script':sha256(Path(__file__)), 'layer_script':sha256(Path(__file__).with_name('export09b_layer_probe.py')),
                'config':sha256(SOURCE/'upstream-bf16/config.json'),'adapter':sha256(adapter_file)}
    receipt = out/'manifest.json'
    if receipt.exists():
        manifest = json.loads(receipt.read_text())
        assert manifest['identity'] == identity, 'Source identity changed; use a new export version'
    else:
        manifest = {'identity':identity,'status':'exporting','scope':'text-only custom ORT prototype, not a public solver',
                    'base_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','adapter':'Reliable01 step60, unselected',
                    'quantization':{'bits':4,'block_size':32,'symmetric':True,'accuracy_level':0},
                    'config':config.to_dict(),'artifacts':{},'started':time.time()}
    def save():atomic_write(receipt,json.dumps(manifest,indent=2)+'\n')
    def done(name):
        if name not in manifest['artifacts']:return False
        assert sha256(out/name)==manifest['artifacts'][name]['sha256'],name
        return True
    def record(name):
        manifest['artifacts'][name]={'sha256':sha256(out/name),'bytes':(out/name).stat().st_size}
        save();print(name,flush=True)
    def tensor(key):
        with safe_open(SOURCE/'upstream-bf16'/index[key],framework='pt') as f:return f.get_tensor(key)
    def export(model, sample, names, outputs, name, dynamic=None):
        if done(name):return
        temporary=out/(name+'.float32.tmp.onnx')
        torch.onnx.export(model.eval(),sample,str(temporary),input_names=names,output_names=outputs,
                          opset_version=18,dynamo=False,dynamic_axes=dynamic,do_constant_folding=True)
        q=MatMulNBitsQuantizer(str(temporary),block_size=32,is_symmetric=True,accuracy_level=0)
        q.process();q.model.save_model_to_file(str(out/name),use_external_data_format=False)
        del q;temporary.unlink();record(name)
    save()
    for idx,kind in enumerate(config.layer_types):
        name=f'layer-{idx:02d}.onnx'
        if done(name):continue
        prefix=f'model.language_model.layers.{idx}.'
        with torch.device('meta'):layer=Qwen3_5DecoderLayer(config,idx)
        weights={k[len(prefix):]:tensor(k).float() for k in index if k.startswith(prefix)}
        layer.load_state_dict(weights,strict=True,assign=True)
        if idx>=29:
            with safe_open(adapter_file,framework='pt') as f:
                for n in ['gate_proj','up_proj','down_proj']:
                    p=f'base_model.model.{prefix}mlp.{n}.'
                    delta=f.get_tensor(p+'lora_B.weight').float() @ f.get_tensor(p+'lora_A.weight').float()*2
                    getattr(layer.mlp,n).weight.add_(delta)
            del delta
        x=torch.zeros(1,1,config.hidden_size)
        if kind=='linear_attention':
            a=layer.linear_attn
            model=RecurrentLayer(layer)
            sample=(x,torch.zeros(1,a.conv_dim,a.conv_kernel_size),torch.zeros(1,a.num_v_heads,a.head_k_dim,a.head_v_dim))
            export(model,sample,['hidden','conv','state'],['hidden_out','conv_out','state_out'],name)
        else:
            model=AttentionLayer(layer)
            rd=int(config.head_dim*config.rope_parameters['partial_rotary_factor'])
            sample=(x,torch.zeros(1,config.num_key_value_heads,1,config.head_dim),torch.zeros(1,config.num_key_value_heads,1,config.head_dim),torch.ones(1,1,rd),torch.zeros(1,1,rd))
            dynamic={n:{2:'past_length'} for n in ['key','value']};dynamic.update({n:{2:'present_length'} for n in ['key_out','value_out']})
            export(model,sample,['hidden','key','value','cos','sin'],['hidden_out','key_out','value_out'],name,dynamic)
        del model,layer,weights,sample;gc.collect()
    rows=8192
    key='model.language_model.embed_tokens.weight'
    with safe_open(SOURCE/'upstream-bf16'/index[key],framework='pt') as f:
        sliced=f.get_slice(key)
        for start in range(0,config.vocab_size,rows):
            name=f'embed-{start//rows:02d}.bf16'
            if done(name):continue
            original=sliced[start:min(start+rows,config.vocab_size)]
            assert original.dtype == torch.bfloat16
            original.contiguous().view(torch.uint16).numpy().tofile(out/name);record(name)
    norm=Qwen3_5RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    norm.weight.copy_(tensor('model.language_model.norm.weight').float())
    rows_head=16384
    with safe_open(SOURCE/'upstream-bf16'/index['lm_head.weight'],framework='pt') as f:
        sliced=f.get_slice('lm_head.weight')
        for start in range(0,config.vocab_size,rows_head):
            name=f'head-{start//rows_head:02d}.onnx'
            if done(name):continue
            model=Head(norm,sliced[start:min(start+rows_head,config.vocab_size)].float())
            export(model,(torch.zeros(1,1,config.hidden_size),),['hidden'],['logits'],name)
            del model;gc.collect()
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':INSTRUCTION+'\n\n'+QUESTION}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer(prompt,add_special_tokens=False).input_ids
    atomic_write(out/'fixture.json',json.dumps({'prompt':prompt,'input_ids':ids,'max_new_tokens':128,'expected_answer':9},indent=2)+'\n')
    manifest.update(status='export-complete-native-and-browser-generation-pending',embedding_rows_per_shard=rows,head_rows_per_shard=rows_head,
                    elapsed_seconds=time.time()-manifest['started'],bytes_total=sum(x['bytes'] for x in manifest['artifacts'].values()))
    save();print(manifest['status'],manifest['bytes_total'],flush=True)


if __name__=='__main__':main()
