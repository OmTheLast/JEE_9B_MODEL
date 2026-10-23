"""Export original MLX affine weights without requantization, one decoder layer at a time."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,Qwen3_5TextRotaryEmbedding
from export09b_layer_probe import ROOT,SOURCE,RecurrentLayer,AttentionLayer,error
from checkpoint import atomic_write

OUT=ROOT/'data/jee/training/next01/export09b04'
CONTROL=ROOT/'data/jee/training/next01/export09b03'
MLX=Path(json.loads((ROOT/'data/jee/training/setup01/model-download.json').read_text())['path'])

def save(path,value):atomic_write(path,json.dumps(value,indent=2)+'\n')

def load_prefix(prefix):
    result={}
    for path in sorted(MLX.glob('*.safetensors')):
        with safe_open(path,framework='pt') as f:
            for name in f.keys():
                if name.startswith(prefix):
                    t=f.get_tensor(name);result[name[len(prefix):]]=t.numpy() if t.dtype==torch.uint32 else t.float().numpy()
    return result

def dequant(w,s,b):
    codes=((w[...,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(w.shape[0],s.shape[1],64)
    return (codes.astype(np.float32)*s[...,None]+b[...,None]).reshape(w.shape[0],-1)

class Linear(torch.nn.Module):
    def __init__(self,weight,a=None,b=None):
        super().__init__();self.weight=torch.nn.Parameter(weight,requires_grad=False)
        if a is not None:
            self.register_buffer('lora_a',a);self.register_buffer('lora_b',b)
    def forward(self,x):
        y=torch.nn.functional.linear(x,self.weight)
        if hasattr(self,'lora_a'):y=y+torch.nn.functional.linear(torch.nn.functional.linear(x,self.lora_a),self.lora_b)*2
        return y

class MLXRecurrent(RecurrentLayer):
    def forward(self,x,conv,state):
        layer=self.layer;a=layer.linear_attn;zinput=layer.input_layernorm(x)
        qkv=a.in_proj_qkv(zinput).transpose(1,2);conv=torch.cat((conv[:,:,1:],qkv),dim=-1)
        qkv=torch.nn.functional.silu((conv*a.conv1d.weight.squeeze(1)).sum(-1))
        q,k,v=torch.split(qkv,[a.key_dim,a.key_dim,a.value_dim],dim=-1)
        q=q.reshape(1,a.num_k_heads,a.head_k_dim).repeat_interleave(a.num_v_heads//a.num_k_heads,1)
        k=k.reshape(1,a.num_k_heads,a.head_k_dim).repeat_interleave(a.num_v_heads//a.num_k_heads,1)
        v=v.reshape(1,a.num_v_heads,a.head_v_dim)
        # MLX uses RMS norm here: mean epsilon becomes head_dim * epsilon in sum form.
        q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+a.head_k_dim*1e-6)/a.head_k_dim**.5
        k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+a.head_k_dim*1e-6)
        beta=a.in_proj_b(zinput).sigmoid().reshape(1,a.num_v_heads,1)
        decay=(-a.A_log.exp()*torch.nn.functional.softplus(a.in_proj_a(zinput)+a.dt_bias)).exp()
        state=state*decay.reshape(1,a.num_v_heads,1,1)
        delta=(v-(state*k.unsqueeze(-1)).sum(-2))*beta;state=state+k.unsqueeze(-1)*delta.unsqueeze(-2)
        y=(state*q.unsqueeze(-1)).sum(-2);z=a.in_proj_z(zinput).reshape(-1,a.head_v_dim)
        y=a.norm(y.reshape(-1,a.head_v_dim),z).reshape(1,1,a.value_dim)
        x=x+a.out_proj(y);return x+layer.mlp(layer.post_attention_layernorm(x)),conv,state

def load_layer(idx,adapted=True):
    cfg=AutoConfig.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True).text_config;cfg._attn_implementation='eager'
    raw=load_prefix(f'language_model.model.layers.{idx}.');quant={};weights={}
    for name,value in raw.items():
        if name.endswith(('.scales','.biases')):continue
        if value.dtype==np.uint32:
            stem=name[:-len('.weight')];s=raw[stem+'.scales'];b=raw[stem+'.biases'];quant['layer.'+name]=(value,s,b)
            weights[name]=torch.from_numpy(dequant(value,s,b))
        else:
            if name.endswith('conv1d.weight'):value=value.transpose(0,2,1).copy()
            if name.endswith(('input_layernorm.weight','post_attention_layernorm.weight','q_norm.weight','k_norm.weight')):value=value-1.0
            weights[name]=torch.from_numpy(value)
    with torch.device('meta'):layer=Qwen3_5DecoderLayer(cfg,idx)
    layer.load_state_dict(weights,strict=True,assign=True);layer.eval();del weights
    with safe_open(SOURCE/'peft-reliable01-step060-v1/adapter_model.safetensors',framework='pt') as f:
        for name,module in list(layer.named_modules()):
            if not isinstance(module,torch.nn.Linear):continue
            a=b=None
            if adapted and idx>=29 and name.startswith('mlp.'):
                p=f'base_model.model.model.language_model.layers.{idx}.{name}.'
                a=f.get_tensor(p+'lora_A.weight').float();b=f.get_tensor(p+'lora_B.weight').float()
            parent,_,child=name.rpartition('.');setattr(layer.get_submodule(parent),child,Linear(module.weight,a,b))
    return cfg,layer,quant

def replace_quantized_matmuls(model,quant):
    initial={v.name:v for v in model.graph.initializer};producers={name:node for node in model.graph.node for name in node.output};nodes=[];added=[];replaced=[]
    for node in model.graph.node:
        source=None
        if node.op_type=='MatMul' and len(node.input)==2:
            trans=producers.get(node.input[1])
            if trans is not None and trans.op_type=='Transpose' and trans.input[0] in quant:source=trans.input[0]
        if source is None:nodes.append(node);continue
        w,s,b=quant[source];n,groups=s.shape;k=groups*64;p=node.output[0]+'__affine_'
        values={'packed':w.astype('<u4').view(np.uint8).reshape(n,groups,32),'scales':s,'zero':np.zeros(n*((groups+1)//2),np.uint8),'shape':np.array([1,1,groups,64],np.int64),'axes':np.array([-1],np.int64),'bias':b.T.copy()}
        added.extend(nh.from_array(v,p+name) for name,v in values.items())
        nodes.extend([h.make_node('MatMulNBits',[node.input[0],p+'packed',p+'scales',p+'zero'],[p+'scaled'],domain='com.microsoft',K=k,N=n,bits=4,block_size=64,accuracy_level=0),h.make_node('Reshape',[node.input[0],p+'shape'],[p+'grouped']),h.make_node('ReduceSum',[p+'grouped',p+'axes'],[p+'sums'],keepdims=0),h.make_node('MatMul',[p+'sums',p+'bias'],[p+'offset']),h.make_node('Add',[p+'scaled',p+'offset'],list(node.output))]);replaced.append(source)
    assert set(replaced)==set(quant),('Missing original weights',set(quant)-set(replaced))
    needed={v.name for v in model.graph.output};kept=[]
    for node in reversed(nodes):
        if needed.intersection(node.output):kept.append(node);needed.update(node.input)
    del model.graph.node[:];model.graph.node.extend(reversed(kept))
    del model.graph.initializer[:];model.graph.initializer.extend(v for v in [*initial.values(),*added] if v.name in needed)
    if not any(v.domain=='com.microsoft' for v in model.opset_import):model.opset_import.append(h.make_opsetid('com.microsoft',1))
    onnx.checker.check_model(model);return replaced

@torch.inference_mode()
def export_layer(idx):
    dest=OUT/f'layer{idx:02d}-v1';dest.mkdir(exist_ok=True)
    identity={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((SOURCE/'peft-reliable01-step060-v1/adapter_model.safetensors').read_bytes()).hexdigest()}
    if (dest/'result.json').exists():
        old=json.loads((dest/'result.json').read_text());assert old['identity']==identity;return
    cfg,layer,quant=load_layer(idx);rec=cfg.layer_types[idx]=='linear_attention';wrapper=MLXRecurrent(layer) if rec else AttentionLayer(layer)
    x=torch.from_numpy(np.load(CONTROL/'addition/trace-adapted-bf16.npz')[f'layer{idx:02d}-in'][:,-1:].copy())
    state=[torch.zeros(1,8192,4),torch.zeros(1,32,128,128)] if rec else [torch.zeros(1,4,0,256),torch.zeros(1,4,0,256)]
    sample=(x,*state) if rec else (x,*state,torch.ones(1,1,64),torch.zeros(1,1,64))
    names=['hidden','conv','state'] if rec else ['hidden','key','value','cos','sin'];outputs=['hidden_out','conv_out','state_out'] if rec else ['hidden_out','key_out','value_out']
    dynamic=None if rec else {n:{2:'past_length'} for n in ['key','value']}
    if dynamic:dynamic.update({n:{2:'present_length'} for n in ['key_out','value_out']})
    expected=[v.numpy().copy() for v in wrapper(*sample)]
    temporary=dest/'dequantized.tmp.onnx'
    torch.onnx.export(wrapper,sample,str(temporary),input_names=names,output_names=outputs,opset_version=18,dynamo=False,do_constant_folding=False,dynamic_axes=dynamic)
    graph=onnx.load(temporary);replaced=replace_quantized_matmuls(graph,quant);onnx.save(graph,dest/'layer.onnx');del graph;temporary.unlink()
    feed={n:v.numpy() for n,v in zip(names,sample)};del wrapper,layer,quant;gc.collect()
    opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session=ort.InferenceSession(str(dest/'layer.onnx'),opt,providers=['CPUExecutionProvider']);actual=session.run(None,feed)
    checks=[error(a,b) for a,b in zip(actual,expected)];passed=all(np.allclose(a,b,atol=5e-4,rtol=5e-4) for a,b in zip(actual,expected))
    np.savez(dest/'control.npz',**feed,**{n:v for n,v in zip(outputs,expected)})
    result={'identity':identity,'layer':idx,'replaced_projections':replaced,'onnx_vs_dequantized_fp32':checks,'passed':passed,'file_bytes':(dest/'layer.onnx').stat().st_size,'scope':'single token at zero state; original MLX full-layer comparison still required'}
    save(dest/'result.json',result);print(json.dumps(result),flush=True)
    assert passed

if __name__=='__main__':
    torch.set_num_threads(4);ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,choices=[0,3,29,31],required=True);args=ap.parse_args();OUT.mkdir(exist_ok=True);export_layer(args.layer)
