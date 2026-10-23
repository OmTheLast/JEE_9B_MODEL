"""Compare native FP32, custom FP32 and q4 on identical real image-prompt layer inputs."""
import argparse,gc,json,time
import numpy as np
import torch
import onnxruntime as ort
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,Qwen3_5TextRotaryEmbedding
from export09b_layer_probe import SOURCE,OUT as OLD,RecurrentLayer,AttentionLayer,error
from export09b03_controls import OUT,CASES,save

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,choices=[0,3,29,31],required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    idx=args.layer;cfg=AutoConfig.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True).text_config;cfg._attn_implementation='eager'
    with torch.device('meta'):layer=Qwen3_5DecoderLayer(cfg,idx)
    prefix=f'model.language_model.layers.{idx}.';weights={}
    index=json.loads((SOURCE/'upstream-bf16/model.safetensors.index.json').read_text())['weight_map']
    for name,shard in index.items():
        if name.startswith(prefix):
            with safe_open(SOURCE/'upstream-bf16'/shard,framework='pt') as f:weights[name[len(prefix):]]=f.get_tensor(name).float()
    layer.load_state_dict(weights,strict=True,assign=True);layer.eval();del weights
    if idx>=29:
        with safe_open(SOURCE/'peft-reliable01-step060-v1/adapter_model.safetensors',framework='pt') as f:
            for name in ['gate_proj','up_proj','down_proj']:
                key='base_model.model.'+prefix+'mlp.'+name+'.'
                getattr(layer.mlp,name).weight.add_(f.get_tensor(key+'lora_B.weight').float()@f.get_tensor(key+'lora_A.weight').float()*2)
    rec=cfg.layer_types[idx]=='linear_attention';wrapper=RecurrentLayer(layer) if rec else AttentionLayer(layer)
    rotary=Qwen3_5TextRotaryEmbedding(cfg)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session=ort.InferenceSession(str(OLD/f'text-q4-v2/layer-{idx:02d}.onnx'),opts,providers=['CPUExecutionProvider'])
    for case in CASES:
        dest=OUT/case;resultfile=dest/f'decoder-layer{idx:02d}.json'
        if resultfile.exists():continue
        start=time.time();trace=np.load(dest/'trace-adapted-bf16.npz');x=torch.from_numpy(trace[f'layer{idx:02d}-in'].copy())
        positions=torch.from_numpy(np.load(dest/'processor.npz')['position_ids'])
        cos,sin=rotary(x,positions);n=x.shape[1]
        mask=None if rec else torch.triu(torch.full((1,1,n,n),torch.finfo(torch.float32).min),diagonal=1)
        native=layer(x,position_embeddings=(cos,sin),attention_mask=mask).numpy()
        states=[torch.zeros(1,8192,4),torch.zeros(1,32,128,128)] if rec else [torch.zeros(1,4,0,256),torch.zeros(1,4,0,256)]
        qstates=[v.numpy().copy() for v in states];custom=[];quant=[]
        for t in range(n):
            h=x[:,t:t+1];sample=[h,*states]+([] if rec else [cos[:,t:t+1],sin[:,t:t+1]])
            actual=wrapper(*sample);custom.append(actual[0].numpy());states=list(actual[1:])
            feed={'hidden':h.numpy()}
            feed.update(dict(zip(['conv','state'] if rec else ['key','value'],qstates)))
            if not rec:feed.update(cos=cos[:,t:t+1].numpy(),sin=sin[:,t:t+1].numpy())
            q=session.run(None,feed);quant.append(q[0]);qstates=q[1:]
        custom=np.concatenate(custom,axis=1);quant=np.concatenate(quant,axis=1)
        np.savez(dest/f'decoder-layer{idx:02d}.npz',native_fp32=native,custom_fp32=custom,q4=quant,custom_state0=states[0].numpy(),custom_state1=states[1].numpy(),q4_state0=qstates[0],q4_state1=qstates[1])
        result={'layer':idx,'tokens':n,'precision':'native/custom FP32, exact same BF16-control input values promoted to FP32; folded adapter','native_vs_custom':error(custom,native),'q4_vs_custom':error(quant,custom),'native_fp32_vs_recorded_bf16':error(native,trace[f'layer{idx:02d}-out']),'last_token_native_vs_custom':error(custom[:,-1],native[:,-1]),'last_token_q4_vs_custom':error(quant[:,-1],custom[:,-1]),'q4_state_errors':[error(a,b.numpy()) for a,b in zip(qstates,states)],'elapsed_seconds':time.time()-start,'scope':'layer-local comparison, not full-model FP32 end-to-end or JEE accuracy'}
        result['native_custom_allclose']=bool(np.allclose(custom,native,atol=5e-4,rtol=5e-4))
        save(resultfile,result);print(case,json.dumps(result),flush=True)

if __name__=='__main__':main()
