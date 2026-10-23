"""Native CPU generation control for the custom text-only ONNX export."""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from export09b_layer_probe import SOURCE,OUT
from checkpoint import atomic_write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-new-tokens',type=int,default=128,choices=[64,128,256,384]);parser.add_argument('--image',action='store_true');args=parser.parse_args()
    root=OUT/'text-q4-v2';label='image' if args.image else 'generation';output=root/f'native-{label}-{args.max_new_tokens}.json'
    if output.exists():raise RuntimeError('Result already exists; preserve the prior run')
    manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    assert manifest['status'].startswith('export-complete')
    fixture=json.loads(((OUT/'image-arithmetic-v1' if args.image else root)/'fixture.json').read_text())
    features=np.fromfile(OUT/'image-arithmetic-v1/features.f32',dtype=np.float32).reshape(-1,4096) if args.image else None
    image_cursor=0
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True)
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sessions=[];states=[];started=time.time()
    result={'status':'loading','scope':'9B actual adapted text-only custom ONNX CPU control','generated_ids':[],
            'prompt':fixture['prompt'],'input_ids':fixture['input_ids'],'start':started}
    result['max_new_tokens']=args.max_new_tokens
    def save():atomic_write(root/f'native-{label}-progress-{args.max_new_tokens}.json',json.dumps(result,indent=2)+'\n')
    save()
    for i,kind in enumerate(cfg['layer_types']):
        sessions.append(ort.InferenceSession(str(root/f'layer-{i:02d}.onnx'),options,providers=['CPUExecutionProvider']))
        states.append([np.zeros((1,8192,4),np.float32),np.zeros((1,32,128,128),np.float32)] if kind=='linear_attention' else [np.zeros((1,4,0,256),np.float32),np.zeros((1,4,0,256),np.float32)])
    heads=[ort.InferenceSession(str(p),options,providers=['CPUExecutionProvider']) for p in sorted(root.glob('head-*.onnx'))]
    result['load_seconds']=time.time()-started;result['status']='running';save()
    rowcount=manifest['embedding_rows_per_shard'];embed={}
    def embedding(token):
        shard=token//rowcount
        if shard not in embed:embed[shard]=np.memmap(root/f'embed-{shard:02d}.bf16',dtype='<u2',mode='r').reshape(-1,4096)
        return (np.array(embed[shard][token%rowcount],dtype=np.uint32)<<16).view(np.float32).reshape(1,1,4096)
    ids=fixture['input_ids'].copy();promptlen=len(ids);position=0
    while position<len(ids):
        if args.image and position<promptlen and ids[position]==fixture['image_token_id']:
            hidden=features[image_cursor].reshape(1,1,4096);image_cursor+=1
        else:hidden=embedding(ids[position])
        positions=[fixture['position_ids'][a][position] if position<promptlen else position+fixture['rope_delta'] for a in range(3)] if args.image else [position]*3
        vector=np.full(32,positions[0],dtype=np.float32);vector[1:33:3]=positions[1];vector[2:30:3]=positions[2]
        freqs=vector*(1/(np.float32(1e7)**(np.arange(0,64,2,dtype=np.float32)/64)))
        freqs=np.concatenate((freqs,freqs)).reshape(1,1,64);cos=np.cos(freqs);sin=np.sin(freqs)
        for i,(session,kind) in enumerate(zip(sessions,cfg['layer_types'])):
            if kind=='linear_attention':feed={'hidden':hidden,'conv':states[i][0],'state':states[i][1]}
            else:feed={'hidden':hidden,'key':states[i][0],'value':states[i][1],'cos':cos,'sin':sin}
            hidden,s0,s1=session.run(None,feed);states[i]=[s0,s1]
        position+=1
        if position>=promptlen:
            logits=np.concatenate([s.run(None,{'hidden':hidden})[0].ravel() for s in heads])
            token=int(logits.argmax());result['generated_ids'].append(token)
            if len(result['generated_ids'])==1:
                top=np.argsort(logits)[-5:][::-1];result['first_top5']=[{'id':int(k),'logit':float(logits[k])} for k in top]
            result['text']=tokenizer.decode(result['generated_ids'],skip_special_tokens=True)
            result['elapsed_seconds']=time.time()-started;save()
            if token==cfg['eos_token_id'] or len(result['generated_ids'])>=args.max_new_tokens:break
            ids.append(token)
        elif position%10==0:
            result['prefill_tokens_done']=position;save()
    result['stop_reason']='eos' if result['generated_ids'][-1]==cfg['eos_token_id'] else 'length'
    result['status']='completed';result['elapsed_seconds']=time.time()-started
    atomic_write(output,json.dumps(result,indent=2)+'\n');save();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
