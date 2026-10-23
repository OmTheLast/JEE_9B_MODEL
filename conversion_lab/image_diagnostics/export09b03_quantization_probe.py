"""Controlled layer-only quantization alternatives on preserved real image inputs."""
import gc,json,logging,time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from export09b03_controls import OUT,OLD,CASES,save
from export09b_layer_probe import error

def main():
    logging.getLogger().setLevel(logging.ERROR)
    dest=OUT/'quantization';dest.mkdir(exist_ok=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    for idx,version in [(29,'v3'),(31,'v2')]:
        for label,bits,symmetric in [('q4-asymmetric',4,False),('q8-symmetric',8,True)]:
            resultpath=dest/f'layer{idx:02d}-{label}.json'
            if resultpath.exists():continue
            path=dest/f'layer{idx:02d}-{label}.onnx';start=time.time()
            if not path.exists():
                q=MatMulNBitsQuantizer(str(OLD/f'layer{idx:02d}-{version}/layer.fp32.onnx'),bits=bits,block_size=32,is_symmetric=symmetric,accuracy_level=0)
                q.process();q.model.save_model_to_file(str(path),use_external_data_format=False);del q;gc.collect()
            session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']);results={}
            for case in CASES:
                root=OUT/case;trace=np.load(root/'trace-adapted-bf16.npz');x=trace[f'layer{idx:02d}-in'];pos=np.load(root/'processor.npz')['position_ids'][:,0,:]
                rec=idx==29;state=[np.zeros((1,8192,4),np.float32),np.zeros((1,32,128,128),np.float32)] if rec else [np.zeros((1,4,0,256),np.float32),np.zeros((1,4,0,256),np.float32)];outputs=[]
                for t in range(x.shape[1]):
                    feed={'hidden':x[:,t:t+1]};feed.update(dict(zip(['conv','state'] if rec else ['key','value'],state)))
                    if not rec:
                        v=np.full(32,pos[0,t],np.float32);v[1:33:3]=pos[1,t];v[2:30:3]=pos[2,t]
                        freq=v/(np.float32(1e7)**(np.arange(0,64,2,dtype=np.float32)/64));freq=np.concatenate([freq,freq]).reshape(1,1,64);feed.update(cos=np.cos(freq),sin=np.sin(freq))
                    y,*state=session.run(None,feed);outputs.append(y)
                actual=np.concatenate(outputs,axis=1);ref=np.load(root/f'decoder-layer{idx:02d}.npz')['custom_fp32']
                results[case]={'vs_fp32':error(actual,ref),'last_token_vs_fp32':error(actual[:,-1],ref[:,-1])}
                np.save(dest/f'layer{idx:02d}-{label}-{case}.npy',actual)
            del session;gc.collect()
            result={'layer':idx,'bits':bits,'symmetric':symmetric,'block_size':32,'accuracy_level':0,'conditions':results,'file_bytes':path.stat().st_size,'elapsed_seconds':time.time()-start,'scope':'layer-only CPU comparison; no full-model answer or browser support claim'}
            save(resultpath,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
