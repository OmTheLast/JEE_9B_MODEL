"""Isolate vision precision and full-model precision on two synthetic images."""
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
import argparse,json,time
import numpy as np
import torch
from transformers import AutoModelForImageTextToText
from peft import PeftModel
from export09b03_controls import OUT,SOURCE,CASES,prepared,save

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['bf16-fp32vision','fp32','fp32-cpu'],required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    device='cpu' if args.mode=='fp32-cpu' else 'mps'
    if device=='mps':torch.mps.set_per_process_memory_fraction(0.9)
    model=AutoModelForImageTextToText.from_pretrained(SOURCE/'upstream-bf16',dtype=torch.float32 if args.mode.startswith('fp32') else torch.bfloat16,device_map={'':device},local_files_only=True).eval()
    model=PeftModel.from_pretrained(model,SOURCE/'peft-reliable01-step060-v1',is_trainable=False).eval()
    for case in CASES:
        dest=OUT/case;path=dest/(args.mode+'.json')
        if path.exists():continue
        p,prompt,v=prepared(case);inputs={k:t.to(device) for k,t in v.items()}
        def hook(module,inputs,output):
            output.pooler_output=torch.from_numpy(np.load(dest/'vision-native-fp32.npy')).to(device='mps',dtype=output.pooler_output.dtype)
            return output
        handle=model.get_base_model().model.visual.register_forward_hook(hook) if args.mode=='bf16-fp32vision' else None
        start=time.time();output=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
        ids=output[0,v['input_ids'].shape[1]:].cpu().tolist()
        if handle:handle.remove()
        r={'mode':args.mode,'adapter':'Reliable01 step60','generated_ids':ids,'text':p.tokenizer.decode(ids,skip_special_tokens=True),'seconds':time.time()-start,'allocated_memory_bytes':torch.mps.current_allocated_memory() if device=='mps' else None,'driver_allocated_memory_bytes':torch.mps.driver_allocated_memory() if device=='mps' else None,'limit_note':'sampled after generation, not peak memory'}
        save(path,r);print(case,r,flush=True)

if __name__=='__main__':main()
