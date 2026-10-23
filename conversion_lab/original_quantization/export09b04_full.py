"""Common original-MLX q4 decoder with explicit rank-4 adapter inputs and checkpoint catalog."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
from export09b04_original import OUT,ROOT,SOURCE,MLX,load_layer,load_prefix,dequant,Linear,MLXRecurrent,AttentionLayer,replace_quantized_matmuls,save,error

PACKAGE=OUT/'package-v1'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(4*1024*1024):digest.update(data)
    return digest.hexdigest()

class Head(torch.nn.Module):
    def __init__(self,norm,linear):super().__init__();self.norm=norm;self.linear=linear
    def forward(self,hidden):return self.linear(self.norm(hidden))

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(431)
    assert json.loads((OUT/'original-layer-parity-v1.json').read_text())['passed']
    PACKAGE.mkdir(exist_ok=True)
    identity={'script':sha(Path(__file__)),'layer_script':sha(Path(__file__).with_name('export09b04_original.py')),'mlx_config':sha(MLX/'config.json'),'upstream_config':sha(SOURCE/'upstream-bf16/config.json')}
    receipt=PACKAGE/'manifest.json'
    if receipt.exists():manifest=json.loads(receipt.read_text());assert manifest['identity']==identity,'Source changed; new package version required'
    else:manifest={'status':'exporting','identity':identity,'source':'mlx-community/Qwen3.5-9B-4bit','source_revision':'8b2b98c00a6b4d291155e4890773ca8f769aee53','quantization':{'bits':4,'group_size':64,'mode':'original affine codes/scales/offsets preserved; FP32 arithmetic'},'artifacts':{},'layer_checks':{},'adapters':{},'started':time.time()}
    def commit():save(receipt,manifest)
    def done(name):
        if name not in manifest['artifacts']:return False
        assert sha(PACKAGE/name)==manifest['artifacts'][name]['sha256'],name
        return True
    def record(name):manifest['artifacts'][name]={'sha256':sha(PACKAGE/name),'bytes':(PACKAGE/name).stat().st_size};commit();print(name,flush=True)
    def export(module,sample,names,outputs,name,quant,dynamic=None,lora_inputs=False):
        temporary=PACKAGE/(name+'.temporary.onnx')
        torch.onnx.export(module.eval(),sample,str(temporary),input_names=names,output_names=outputs,opset_version=18,dynamo=False,do_constant_folding=False,dynamic_axes=dynamic,keep_initializers_as_inputs=lora_inputs)
        graph=onnx.load(temporary);replace_quantized_matmuls(graph,quant)
        # Six REQUIRED adapter inputs. Distinct export fixtures prevent zero-tensor
        # deduplication; removing their initializers prevents accidental hidden adapters.
        if lora_inputs:
            allowed=set(names)|{i.name for i in graph.graph.initializer if i.name.endswith(('.lora_a','.lora_b'))}
            inputs=[i for i in graph.graph.input if i.name in allowed];del graph.graph.input[:];graph.graph.input.extend(inputs)
            initial=[i for i in graph.graph.initializer if not i.name.endswith(('.lora_a','.lora_b'))]
            del graph.graph.initializer[:];graph.graph.initializer.extend(initial)
            assert len(inputs)==len(names)+6
        onnx.checker.check_model(graph);onnx.save(graph,PACKAGE/name);del graph;temporary.unlink();gc.collect()
    commit()
    for idx in range(32):
        name=f'layer-{idx:02d}.onnx'
        if done(name):continue
        cfg,layer,quant=load_layer(idx,adapted=False);manifest['config']=cfg.to_dict()
        for projection in ['gate_proj','up_proj','down_proj']:
            linear=getattr(layer.mlp,projection);n,k=linear.weight.shape
            linear.register_buffer('lora_a',torch.randn(4,k)*.01);linear.register_buffer('lora_b',torch.randn(n,4)*.01)
        rec=cfg.layer_types[idx]=='linear_attention';module=MLXRecurrent(layer) if rec else AttentionLayer(layer)
        x=torch.randn(1,1,cfg.hidden_size)*.1
        sample=(x,torch.zeros(1,8192,4),torch.zeros(1,32,128,128)) if rec else (x,torch.zeros(1,4,1,256),torch.zeros(1,4,1,256),torch.ones(1,1,64),torch.zeros(1,1,64))
        names=['hidden','conv','state'] if rec else ['hidden','key','value','cos','sin'];outputs=['hidden_out','conv_out','state_out'] if rec else ['hidden_out','key_out','value_out']
        dynamic=None if rec else {n:{2:'past_length'} for n in ['key','value']}
        if dynamic:dynamic.update({n:{2:'present_length'} for n in ['key_out','value_out']})
        expected=[v.numpy().copy() for v in module(*sample)];feeds={n:v.numpy().copy() for n,v in zip(names,sample)}
        adapter_inputs={n:v.numpy().copy() for n,v in module.named_buffers() if n.endswith(('.lora_a','.lora_b'))}
        feeds.update(adapter_inputs);manifest['adapter_input_shapes']={n:list(v.shape) for n,v in adapter_inputs.items()}
        export(module,sample,names,outputs,name,quant,dynamic,True)
        del module,layer,quant,sample;gc.collect()
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session=ort.InferenceSession(str(PACKAGE/name),opts,providers=['CPUExecutionProvider']);actual=session.run(None,feeds)
        checks=[error(a,b) for a,b in zip(actual,expected)];assert all(c['relative_l2']<1e-4 for c in checks),checks
        manifest['layer_checks'][str(idx)]=checks;del session;gc.collect();record(name)
    # The original quantized embedding values are small enough to decode one row on demand.
    e=load_prefix('language_model.model.embed_tokens.');rows=8192
    for start in range(0,manifest['config']['vocab_size'],rows):
        for field,suffix,dtype in [('weight','u32','<u4'),('scales','scales.f32','<f4'),('biases','biases.f32','<f4')]:
            name=f'embed-{start//rows:02d}.{suffix}'
            if not done(name):e[field][start:start+rows].astype(dtype).tofile(PACKAGE/name);record(name)
    del e;gc.collect();manifest['embedding_rows_per_shard']=rows
    raw=load_prefix('language_model.lm_head.');norm_values=load_prefix('language_model.model.norm.')['weight']
    norm=Qwen3_5RMSNorm(4096,eps=manifest['config']['rms_norm_eps']);norm.weight.copy_(torch.from_numpy(norm_values-1.0))
    rows=16384
    for start in range(0,manifest['config']['vocab_size'],rows):
        name=f'head-{start//rows:02d}.onnx'
        if done(name):continue
        w,s,b=[raw[k][start:start+rows].copy() for k in ['weight','scales','biases']]
        module=Head(norm,Linear(torch.from_numpy(dequant(w,s,b))))
        export(module,(torch.zeros(1,1,4096),),['hidden'],['logits'],name,{'linear.weight':(w,s,b)})
        del module,w,s,b;gc.collect();record(name)
    del raw;gc.collect();manifest['head_rows_per_shard']=rows
    history=ROOT/'data/jee/releases/hf-history-v2/public/adapters'
    for directory in sorted(history.iterdir()):
        if not (directory/'jee_adapter.json').exists():continue
        settings=json.loads((directory/'jee_adapter.json').read_text());assert settings['rank']==4 and settings['scale']==2
        catalog={'source_adapter_sha256':sha(directory/'adapter.safetensors'),'settings':settings,'layers':{}}
        (PACKAGE/'adapters'/directory.name).mkdir(parents=True,exist_ok=True)
        with safe_open(directory/'adapter.safetensors',framework='pt') as f:
            for key in f.keys():
                parts=key.split('.');idx=int(parts[parts.index('layers')+1]);projection=parts[parts.index('mlp')+1];suffix=parts[-1]
                assert suffix in ['lora_a','lora_b'],key
                value=f.get_tensor(key).float().numpy().T.copy()
                name=f'adapters/{directory.name}/{idx:02d}-{projection}-{suffix}.f32'
                if not done(name):value.astype('<f4').tofile(PACKAGE/name);record(name)
                catalog['layers'].setdefault(str(idx),{})[f'layer.mlp.{projection}.{suffix}']={'file':name,'dims':list(value.shape)}
        manifest['adapters'][directory.name]=catalog;commit()
    manifest.update(status='export-complete-parity-pending',bytes_total=sum(v['bytes'] for v in manifest['artifacts'].values()),elapsed_seconds=time.time()-manifest['started'])
    commit();print(manifest['status'],manifest['bytes_total'],flush=True)

if __name__=='__main__':main()
