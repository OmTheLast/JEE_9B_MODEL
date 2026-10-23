"""Synthetic image export diagnosis. Each stage writes immutable, resumable evidence."""
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding
from safetensors import safe_open
from peft import PeftModel
import onnxruntime as ort
from checkpoint import atomic_write
from export09b_layer_probe import ROOT, SOURCE, OUT as OLD, error

OUT = ROOT / 'data/jee/training/next01/export09b03'
QUESTION = 'Read the arithmetic in the image. Reply in one line: FINAL_ANSWER: followed by the result.'
CASES = ['addition', 'subtraction']

def save(path, obj):
    atomic_write(path, json.dumps(obj, indent=2)+'\n')

def prepared(case):
    p = AutoProcessor.from_pretrained(SOURCE/'upstream-bf16', local_files_only=True)
    prompt = p.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTION}]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    v = p(text=[prompt], images=[Image.open(OUT/case/'question.png').convert('RGB')], do_resize=False, return_tensors='pt')
    return p, prompt, v

def fixtures():
    cfg = AutoConfig.from_pretrained(SOURCE/'upstream-bf16', local_files_only=True)
    for case in CASES:
        dest=OUT/case;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'fixture.json').exists(): continue
        if case=='addition':
            (dest/'question.png').write_bytes((OLD/'image-arithmetic-v1/question.png').read_bytes())
        else:
            image=Image.new('RGB',(384,384),'white');d=ImageDraw.Draw(image)
            font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',64)
            d.text((44,145),'8 - 1 = ?',font=font,fill='black');image.save(dest/'question.png')
        p,prompt,v=prepared(case)
        owner=SimpleNamespace(config=cfg);owner.get_vision_position_ids=MethodType(Qwen3_5Model.get_vision_position_ids,owner)
        pos,delta=Qwen3_5Model.get_rope_index(owner,v['input_ids'],v['mm_token_type_ids'],image_grid_thw=v['image_grid_thw'])
        arrays={k:x.numpy() for k,x in v.items()};arrays['position_ids']=pos.numpy()
        np.savez(dest/'processor.npz',**arrays)
        pixels=arrays['pixel_values'].astype(np.float32);pixels.tofile(dest/'pixels.f32')
        # Undo the processor's patch packing to inspect exactly what the encoder sees.
        grid=arrays['image_grid_thw'][0];t,h,w=map(int,grid);merge=2;patch=16
        reconstructed=pixels.reshape(t,h//merge,w//merge,merge,merge,3,2,patch,patch).transpose(0,6,5,1,3,7,2,4,8).reshape(t*2,3,h*patch,w*patch)[0]
        reconstructed=np.clip((reconstructed.transpose(1,2,0)*.5+.5)*255,0,255).round().astype(np.uint8)
        Image.fromarray(reconstructed).save(dest/'reconstructed.png')
        expected=np.array(Image.open(dest/'question.png').convert('RGB'))
        ids=arrays['input_ids'][0].tolist()
        f={'input_ids':ids,'prompt':p.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),'position_ids':pos[:,0,:].tolist(),'rope_delta':int(delta[0,0]),'image_token_id':cfg.image_token_id,'pixel_shape':list(pixels.shape),'feature_shape':[144,4096],'expected_answer':5 if case=='addition' else 7,'image_sha256':hashlib.sha256((dest/'question.png').read_bytes()).hexdigest(),'processor_roundtrip_max_pixel_error':int(np.abs(expected.astype(int)-reconstructed.astype(int)).max())}
        if case=='addition':
            old=json.loads((OLD/'image-arithmetic-v1/fixture.json').read_text())
            f['prior_input_ids_equal']=ids==old['input_ids'];f['prior_positions_equal']=f['position_ids']==old['position_ids']
            f['prior_pixels_equal']=np.array_equal(pixels,np.fromfile(OLD/'image-arithmetic-v1/pixels.f32',np.float32).reshape(pixels.shape))
        save(dest/'fixture.json',f)
        print(case,f['processor_roundtrip_max_pixel_error'],flush=True)

@torch.inference_mode()
def vision():
    torch.set_num_threads(4)
    cfg=AutoConfig.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True).vision_config;cfg._attn_implementation='eager'
    with torch.device('meta'): model=Qwen3_5VisionModel(cfg)
    index=json.loads((SOURCE/'upstream-bf16/model.safetensors.index.json').read_text())['weight_map'];weights={}
    for key,shard in index.items():
        if key.startswith('model.visual.'):
            with safe_open(SOURCE/'upstream-bf16'/shard,framework='pt') as f:weights[key[len('model.visual.'):]]=f.get_tensor(key).float()
    model.load_state_dict(weights,strict=True,assign=True);model.rotary_pos_emb=Qwen3_5VisionRotaryEmbedding(cfg);model.eval()
    del weights
    for case in CASES:
        dest=OUT/case
        if (dest/'vision-fp32.json').exists():continue
        a=np.load(dest/'processor.npz');features=model(torch.from_numpy(a['pixel_values']),torch.from_numpy(a['image_grid_thw'])).pooler_output.numpy()
        np.save(dest/'vision-native-fp32.npy',features)
    del model;gc.collect()
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(OLD/'vision384-v1/vision.fp32.onnx'),opts,providers=['CPUExecutionProvider'])
    for case in CASES:
        dest=OUT/case
        if (dest/'vision-fp32.json').exists():continue
        a=np.load(dest/'processor.npz');actual=session.run(None,{'pixels':a['pixel_values']})[0]
        np.save(dest/'vision-onnx-fp32.npy',actual)
        ref=np.load(dest/'vision-native-fp32.npy');metrics=error(actual,ref)
        save(dest/'vision-fp32.json',{'native_vs_onnx':metrics,'allclose_2e-3':bool(np.allclose(actual,ref,atol=2e-3,rtol=2e-3))})
        print(case,metrics,flush=True)

@torch.inference_mode()
def original():
    torch.set_num_threads(4)
    model=AutoModelForImageTextToText.from_pretrained(SOURCE/'upstream-bf16',dtype=torch.bfloat16,device_map={'':'mps'},local_files_only=True).eval()
    for mode in ['base','adapted']:
        if mode=='adapted':model=PeftModel.from_pretrained(model,SOURCE/'peft-reliable01-step060-v1',is_trainable=False).eval()
        core=model.get_base_model().model if mode=='adapted' else model.model
        for case in CASES:
            dest=OUT/case;resultfile=dest/f'original-{mode}.json'
            if resultfile.exists():continue
            p,prompt,v=prepared(case);f=json.loads((dest/'fixture.json').read_text());assert v['input_ids'][0].tolist()==f['input_ids']
            traces={};handles=[]
            def capture_vision(module,args,output):traces['vision']=output.pooler_output.detach().float().cpu().numpy().copy()
            handles.append(core.visual.register_forward_hook(capture_vision))
            for idx in [0,3,29,31]:
                def hook(module,args,kwargs,output,idx=idx):
                    x=args[0] if args else kwargs['hidden_states'];traces[f'layer{idx:02d}-in']=x.detach().float().cpu().numpy()
                    traces[f'layer{idx:02d}-out']=output.detach().float().cpu().numpy()
                    if idx==31:
                        for label,t in zip(['cos','sin'],kwargs['position_embeddings']):traces[label]=t.detach().float().cpu().numpy()
                handles.append(core.language_model.layers[idx].register_forward_hook(hook,with_kwargs=True))
            inputs={k:x.to('mps') for k,x in v.items()};started=time.time()
            logits=model(**inputs,logits_to_keep=1).logits[0,-1].float().cpu().numpy()
            for h in handles:h.remove()
            traces['logits']=logits;np.savez(dest/f'trace-{mode}-bf16.npz',**traces)
            output=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
            ids=output[0,v['input_ids'].shape[1]:].cpu().tolist()
            top=np.argsort(logits)[-10:][::-1]
            result={'precision':'BF16, MPS','mode':mode,'generated_ids':ids,'text':p.tokenizer.decode(ids,skip_special_tokens=True),'first_top10':[{'id':int(k),'logit':float(logits[k])} for k in top],'elapsed_seconds':time.time()-started,'vision_vs_fp32':error(traces['vision'],np.load(dest/'vision-native-fp32.npy'))}
            save(resultfile,result);print(case,mode,result['text'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fixtures','vision','original']);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);globals()[args.stage]()
