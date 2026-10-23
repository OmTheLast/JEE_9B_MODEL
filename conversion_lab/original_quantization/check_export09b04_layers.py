"""Independent MLX FP32 reference for preserved-weight decoder graphs."""
import json
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load
from mlx_vlm.models.cache import ArraysCache,KVCache
import onnxruntime as ort
from export09b04_original import OUT,CONTROL,MLX,save,error
from run_export09b_mlx_parity import attach_adapter

def main():
    path=OUT/'original-layer-parity-v1.json'
    if path.exists():raise RuntimeError('Preserve completed result')
    mx.set_memory_limit(40*2**30);mx.set_cache_limit(2*2**30)
    model,_=load(str(MLX),trust_remote_code=False);model.freeze();model.eval();attach_adapter(model)
    result={'status':'running','precision':'MLX floating parameters and hidden inputs promoted to FP32; packed codes unchanged; same adapter','checks':[]}
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    for idx in [29,31]:
        layer=model.language_model.layers[idx]
        layer.apply(lambda v:v.astype(mx.float32) if v.dtype in [mx.bfloat16,mx.float16] else v)
        session=ort.InferenceSession(str(OUT/f'layer{idx:02d}-v1/layer.onnx'),options,providers=['CPUExecutionProvider'])
        rec=idx==29
        for case in ['addition','subtraction']:
            x=np.load(CONTROL/case/'trace-adapted-bf16.npz')[f'layer{idx:02d}-in'];pos=np.load(CONTROL/case/'processor.npz')['position_ids']
            cache=ArraysCache(size=2) if rec else KVCache()
            expected=layer(mx.array(x),cache=cache,mask=None if rec else 'causal',position_ids=mx.array(pos));mx.eval(expected)
            expected=np.array(expected)
            state=[np.zeros((1,8192,4),np.float32),np.zeros((1,32,128,128),np.float32)] if rec else [np.zeros((1,4,0,256),np.float32),np.zeros((1,4,0,256),np.float32)]
            outputs=[]
            for t in range(x.shape[1]):
                feed={'hidden':x[:,t:t+1]};feed.update(dict(zip(['conv','state'] if rec else ['key','value'],state)))
                if not rec:
                    v=np.full(32,pos[0,0,t],np.float32);v[1:33:3]=pos[1,0,t];v[2:30:3]=pos[2,0,t]
                    freq=v/(np.float32(1e7)**(np.arange(0,64,2,dtype=np.float32)/64));freq=np.concatenate([freq,freq]).reshape(1,1,64);feed.update(cos=np.cos(freq),sin=np.sin(freq))
                output,*state=session.run(None,feed);outputs.append(output)
            actual=np.concatenate(outputs,axis=1)
            if rec:
                mx.eval(cache[0],cache[1]);refstates=[np.array(cache[0]),np.array(cache[1])]
                actualstates=[state[0][:,:,1:].transpose(0,2,1),state[1].transpose(0,1,3,2)]
            else:
                mx.eval(cache.keys,cache.values);refstates=[np.array(cache.keys)[:,:,:x.shape[1]],np.array(cache.values)[:,:,:x.shape[1]]];actualstates=state
            checks=[error(actual,expected),*[error(a,b) for a,b in zip(actualstates,refstates)]]
            passed=checks[0]['relative_l2']<1e-4 and all(c['relative_l2']<1e-4 for c in checks[1:])
            np.savez(OUT/f'layer{idx:02d}-v1/{case}-mlx-parity.npz',mlx=expected,onnx=actual,mlx_state0=refstates[0],mlx_state1=refstates[1],onnx_state0=actualstates[0],onnx_state1=actualstates[1])
            result['checks'].append({'layer':idx,'case':case,'metrics':checks,'passed':passed});save(OUT/'original-layer-parity-progress.json',result);print(result['checks'][-1],flush=True)
    result['status']='complete';result['passed']=all(c['passed'] for c in result['checks']);save(path,result)

if __name__=='__main__':main()
