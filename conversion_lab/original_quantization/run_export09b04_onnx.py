"""Full original-weight ONNX controls, explicit checkpoint inputs, token checkpoints."""
import json,time,argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from export09b04_original import OUT,CONTROL,SOURCE,save
from export09b04_full import PACKAGE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--modes',nargs='*');args=ap.parse_args()
    manifest=json.loads((PACKAGE/'manifest.json').read_text());cfg=manifest['config'];assert manifest['status']=='export-complete-parity-pending'
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sessions=[ort.InferenceSession(str(PACKAGE/f'layer-{i:02d}.onnx'),opts,providers=['CPUExecutionProvider']) for i in range(32)]
    heads=[ort.InferenceSession(str(p),opts,providers=['CPUExecutionProvider']) for p in sorted(PACKAGE.glob('head-*.onnx'))]
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True);stop_ids={int(cfg['eos_token_id']),int(tokenizer.eos_token_id)}
    root=OUT/'onnx-controls';root.mkdir(exist_ok=True);rows=manifest['embedding_rows_per_shard'];embeds={}
    def embedding(token):
        chunk=token//rows
        if chunk not in embeds:
            embeds[chunk]=[np.memmap(PACKAGE/f'embed-{chunk:02d}.{suffix}',dtype=dtype,mode='r').reshape(-1,width) for suffix,dtype,width in [('u32','<u4',512),('scales.f32','<f4',64),('biases.f32','<f4',64)]]
        w,s,b=[v[token%rows] for v in embeds[chunk]]
        q=((w[:,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(64,64).astype(np.float32)
        return (q*s[:,None]+b[:,None]).reshape(1,1,4096)
    for mode in (args.modes or ['base',*manifest['adapters']]):
        adapters=[{name:np.zeros(shape,np.float32) for name,shape in manifest['adapter_input_shapes'].items()} for _ in range(32)]
        if mode!='base':
            for layer,inputs in manifest['adapters'][mode]['layers'].items():
                for name,meta in inputs.items():adapters[int(layer)][name]=np.fromfile(PACKAGE/meta['file'],dtype='<f4').reshape(meta['dims'])
        for case in ['text','addition','subtraction']:
            path=root/(mode+'--'+case+'.json')
            if path.exists():continue
            image=case!='text';f=json.loads(((CONTROL/case/'fixture.json') if image else (OUT/'text-fixture.json')).read_text())
            features=np.load(OUT/(case+'-mlx-vision-fp32.npy')) if image else None
            states=[[np.zeros((1,8192,4),np.float32),np.zeros((1,32,128,128),np.float32)] if kind=='linear_attention' else [np.zeros((1,4,0,256),np.float32),np.zeros((1,4,0,256),np.float32)] for kind in cfg['layer_types']]
            ids=f['input_ids'].copy();n=len(ids);pos=0;cursor=0;generated=[];start=time.time();result={'status':'running','mode':mode,'case':case,'stop_ids':sorted(stop_ids),'image_features':'original MLX FP32 reference' if image else None}
            while pos<len(ids):
                if image and pos<n and ids[pos]==f['image_token_id']:hidden=features[cursor].reshape(1,1,4096);cursor+=1
                else:hidden=embedding(ids[pos])
                positions=[f['position_ids'][a][pos] if pos<n else pos+f['rope_delta'] for a in range(3)] if image else [pos]*3
                v=np.full(32,positions[0],np.float32);v[1:33:3]=positions[1];v[2:30:3]=positions[2]
                freq=v/(np.float32(1e7)**(np.arange(0,64,2,dtype=np.float32)/64));freq=np.concatenate([freq,freq]).reshape(1,1,64)
                for i,(s,kind) in enumerate(zip(sessions,cfg['layer_types'])):
                    feed=dict(adapters[i],hidden=hidden);feed.update(dict(zip(['conv','state'] if kind=='linear_attention' else ['key','value'],states[i])))
                    if kind!='linear_attention':feed.update(cos=np.cos(freq),sin=np.sin(freq))
                    hidden,s0,s1=s.run(None,feed);states[i]=[s0,s1]
                pos+=1
                if pos>=n:
                    logits=np.concatenate([s.run(None,{'hidden':hidden})[0].ravel() for s in heads]);token=int(logits.argmax());generated.append(token)
                    result.update(generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=True),elapsed_seconds=time.time()-start)
                    if len(generated)==1:
                        top=np.argsort(logits)[-10:][::-1];result['first_top10']=[{'id':int(k),'logit':float(logits[k])} for k in top]
                    save(root/(mode+'--'+case+'-progress.json'),result)
                    if token in stop_ids or len(generated)>=64:break
                    ids.append(token)
                elif pos%20==0:save(root/(mode+'--'+case+'-progress.json'),dict(result,prefill_tokens_done=pos))
            reference=json.loads((OUT/'mlx-controls'/(mode+'--'+case+'.json')).read_text())
            result.update(status='complete',finish_reason='stop' if generated[-1] in stop_ids else 'length',exact_mlx_token_match=generated==reference['generated_ids'])
            save(path,result);print(mode,case,result['exact_mlx_token_match'],result['text'],flush=True)

if __name__=='__main__':main()
