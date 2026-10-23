"""Replay the saved q4 model with controlled image features; no exports overwritten."""
import argparse,json,time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from export09b03_controls import OUT,SOURCE,OLD,CASES,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',choices=['fp32','bf16'],required=True);args=ap.parse_args()
    root=OLD/'text-q4-v2';manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sessions=[ort.InferenceSession(str(root/f'layer-{i:02d}.onnx'),opts,providers=['CPUExecutionProvider']) for i in range(32)]
    heads=[ort.InferenceSession(str(p),opts,providers=['CPUExecutionProvider']) for p in sorted(root.glob('head-*.onnx'))]
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True)
    rows=manifest['embedding_rows_per_shard'];embeds={}
    def embedding(token):
        chunk=token//rows
        if chunk not in embeds:embeds[chunk]=np.memmap(root/f'embed-{chunk:02d}.bf16',dtype='<u2',mode='r').reshape(-1,4096)
        return (np.array(embeds[chunk][token%rows],dtype=np.uint32)<<16).view(np.float32).reshape(1,1,4096)
    for case in CASES:
        dest=OUT/case;name='q4-'+args.features+'vision';path=dest/(name+'.json')
        if path.exists():continue
        f=json.loads((dest/'fixture.json').read_text());features=np.load(dest/'vision-onnx-fp32.npy') if args.features=='fp32' else np.load(dest/'trace-adapted-bf16.npz')['vision']
        states=[[np.zeros((1,8192,4),np.float32),np.zeros((1,32,128,128),np.float32)] if kind=='linear_attention' else [np.zeros((1,4,0,256),np.float32),np.zeros((1,4,0,256),np.float32)] for kind in cfg['layer_types']]
        ids=f['input_ids'].copy();n=len(ids);pos=0;cursor=0;generated=[];traces={};start=time.time();result={'status':'running','feature_source':args.features,'case':case}
        while pos<len(ids):
            if pos<n and ids[pos]==f['image_token_id']:h=features[cursor].reshape(1,1,4096);cursor+=1
            else:h=embedding(ids[pos])
            positions=[f['position_ids'][a][pos] if pos<n else pos+f['rope_delta'] for a in range(3)]
            v=np.full(32,positions[0],np.float32);v[1:33:3]=positions[1];v[2:30:3]=positions[2]
            freq=v/(np.float32(1e7)**(np.arange(0,64,2,dtype=np.float32)/64));freq=np.concatenate([freq,freq]).reshape(1,1,64)
            for i,(s,kind) in enumerate(zip(sessions,cfg['layer_types'])):
                if pos==n-1:traces[f'layer{i:02d}-in']=h.copy()
                feed={'hidden':h};feed.update(dict(zip(['conv','state'] if kind=='linear_attention' else ['key','value'],states[i])))
                if kind!='linear_attention':feed.update(cos=np.cos(freq),sin=np.sin(freq))
                h,s0,s1=s.run(None,feed);states[i]=[s0,s1]
                if pos==n-1:traces[f'layer{i:02d}-out']=h.copy()
            pos+=1
            if pos>=n:
                logits=np.concatenate([s.run(None,{'hidden':h})[0].ravel() for s in heads]);token=int(logits.argmax());generated.append(token)
                if len(generated)==1:
                    traces['logits']=logits;top=np.argsort(logits)[-10:][::-1];result['first_top10']=[{'id':int(k),'logit':float(logits[k])} for k in top]
                    np.savez(dest/(name+'-trace.npz'),**traces)
                result.update(generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=True),elapsed_seconds=time.time()-start)
                save(dest/(name+'-progress.json'),result)
                if token==cfg['eos_token_id'] or len(generated)>=64:break
                ids.append(token)
            elif pos%20==0:save(dest/(name+'-progress.json'),dict(result,prefill_tokens_done=pos))
        result.update(status='complete',stop_reason='eos' if generated[-1]==cfg['eos_token_id'] else 'length');save(path,result);print(case,result['text'],flush=True)

if __name__=='__main__':main()
