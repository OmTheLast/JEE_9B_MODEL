"""Preserve original MLX q4 values using MatMulNBits plus the floating group bias.

One real trained MLP projection only. No full-model or browser claim.
"""
import json
import numpy as np
import torch
import mlx.core as mx
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
import onnxruntime as ort
from safetensors import safe_open
from export09b03_controls import ROOT,OUT,SOURCE,save
from export09b_layer_probe import error

def main():
    dest=OUT/'mlx-affine';dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():raise RuntimeError('Preserve completed probe')
    root=ROOT/'data/jee/training/setup01'
    original=__import__('pathlib').Path(json.loads((root/'model-download.json').read_text())['path'])
    prefix='language_model.model.layers.29.mlp.gate_proj.';weights={}
    for shard in sorted(original.glob('*.safetensors')):
        with safe_open(shard,framework='pt') as f:
            for key in f.keys():
                if key.startswith(prefix):
                    value=f.get_tensor(key);weights[key[len(prefix):]]=value.numpy() if value.dtype==torch.uint32 else value.float().numpy()
    w=weights['weight'];s=weights['scales'];b=weights['biases'];n,groups=s.shape;k=groups*64
    x=np.load(OUT/'addition/trace-adapted-bf16.npz')['layer29-in'][0,[-1]].copy()
    assert x.shape==(1,k)
    # MLX packs consecutive low-to-high nibbles in uint32. ORT stores those pairs in uint8.
    packed=w.astype('<u4').view(np.uint8).reshape(n,groups,32)
    init=[nh.from_array(a,name) for name,a in [('packed',packed),('scales',s),('zero',np.zeros(n*((groups+1)//2),np.uint8)),('group_shape',np.array([-1,groups,64],np.int64)),('axes',np.array([-1],np.int64)),('bias_transpose',b.T.copy())]]
    nodes=[h.make_node('MatMulNBits',['x','packed','scales','zero'],['scaled'],domain='com.microsoft',K=k,N=n,bits=4,block_size=64,accuracy_level=0),h.make_node('Reshape',['x','group_shape'],['grouped']),h.make_node('ReduceSum',['grouped','axes'],['sums'],keepdims=0),h.make_node('MatMul',['sums','bias_transpose'],['offset']),h.make_node('Add',['scaled','offset'],['base'])]
    with safe_open(SOURCE/'peft-reliable01-step060-v1/adapter_model.safetensors',framework='pt') as f:
        p='base_model.model.model.language_model.layers.29.mlp.gate_proj.'
        a=f.get_tensor(p+'lora_A.weight').float().numpy();bb=f.get_tensor(p+'lora_B.weight').float().numpy()*2
    init.extend([nh.from_array(a.T.copy(),'lora_a'),nh.from_array(bb.T.copy(),'lora_b')])
    nodes.extend([h.make_node('MatMul',['x','lora_a'],['lowrank']),h.make_node('MatMul',['lowrank','lora_b'],['delta']),h.make_node('Add',['base','delta'],['adapted'])])
    graph=h.make_graph(nodes,'original-mlx-affine-projection',[h.make_tensor_value_info('x',T.FLOAT,[1,k])],[h.make_tensor_value_info(name,T.FLOAT,[1,n]) for name in ['base','adapted']],init)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',18),h.make_opsetid('com.microsoft',1)],ir_version=10);onnx.checker.check_model(model);onnx.save(model,dest/'projection.onnx')
    mx.set_memory_limit(2*2**30)
    ref=mx.quantized_matmul(mx.array(x),mx.array(w),mx.array(s),mx.array(b),transpose=True,group_size=64,bits=4);mx.eval(ref)
    ref=np.array(ref);ref_adapted=ref+(x@a.T)@bb.T
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(dest/'projection.onnx'),options,providers=['CPUExecutionProvider']);base,adapted=session.run(None,{'x':x})
    np.savez(dest/'control.npz',input=x,mlx_base=ref,mlx_plus_adapter=ref_adapted,onnx_base=base,onnx_adapted=adapted)
    result={'base_error':error(base,ref),'adapted_error':error(adapted,ref_adapted),'original_quantization':{'group_size':64,'bits':4,'mode':'affine'},'precision':'matched FP32 inputs/scales/biases and accumulation; original packed codes unchanged','base_allclose':bool(np.allclose(base,ref,atol=5e-4,rtol=5e-4)),'adapted_allclose':bool(np.allclose(adapted,ref_adapted,atol=5e-4,rtol=5e-4)),'file_bytes':(dest/'projection.onnx').stat().st_size,'scope':'one projection on one real hidden input; CPU ORT vs MLX Metal; browser and full model untested'}
    save(dest/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
