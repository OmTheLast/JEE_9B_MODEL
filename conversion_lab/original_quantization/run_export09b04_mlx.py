"""Stream matched-FP32 original MLX controls for base and archived adapters."""
import json,time
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load,prepare_inputs
from mlx_vlm.generate import stream_generate
from mlx_vlm.trainer.lora_layers import LoRALinear
from export09b04_original import ROOT,OUT,CONTROL,MLX,save,error
from export09b03_controls import prepared

def main():
    root=OUT/'mlx-controls';root.mkdir(exist_ok=True)
    mx.set_memory_limit(40*2**30);mx.set_cache_limit(2*2**30);mx.random.seed(0)
    model,processor=load(str(MLX),trust_remote_code=False);model.freeze();model.eval()
    # Preserve all integer codes and numeric parameter values, change arithmetic precision only.
    model.apply(lambda v:v.astype(mx.float32) if v.dtype in [mx.bfloat16,mx.float16] else v)
    for layer in model.language_model.layers:
        for name in ['gate_proj','up_proj','down_proj']:
            setattr(layer.mlp,name,LoRALinear.from_base(getattr(layer.mlp,name),r=4,scale=2,dropout=0))
    fixtures={}
    text=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':'What is 2 + 3? Reply in one line: FINAL_ANSWER: followed by the result.'}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    fixtures['text']={'prompt':text,'image':None}
    save(OUT/'text-fixture.json',{'prompt':text,'input_ids':processor.tokenizer(text,add_special_tokens=False).input_ids,'expected_answer':5,'max_new_tokens':64})
    for case in ['addition','subtraction']:
        _,prompt,_=prepared(case);image=CONTROL/case/'question.png';fixtures[case]={'prompt':prompt,'image':image}
        p=prepare_inputs(processor,prompts=[prompt],images=[str(image)],add_special_tokens=False,padding=True,return_tensors='mlx')
        features,_=model.vision_tower(p['pixel_values'].astype(mx.float32),p['image_grid_thw']);mx.eval(features);a=np.array(features)
        np.save(OUT/(case+'-mlx-vision-fp32.npy'),a)
        save(OUT/(case+'-vision-parity.json'),{'mlx_fp32_vs_torch_fp32':error(a,np.load(CONTROL/case/'vision-native-fp32.npy'))})
    catalog=ROOT/'data/jee/releases/hf-history-v2/public/adapters'
    for mode in ['base',*[p.name for p in sorted(catalog.iterdir()) if (p/'jee_adapter.json').exists()]]:
        for layer in model.language_model.layers:
            for name in ['gate_proj','up_proj','down_proj']:
                linear=getattr(layer.mlp,name);linear.lora_a=mx.zeros_like(linear.lora_a);linear.lora_b=mx.zeros_like(linear.lora_b)
        if mode!='base':model.load_weights(list(mx.load(str(catalog/mode/'adapter.safetensors')).items()),strict=False)
        model.eval()
        for case,f in fixtures.items():
            path=root/(mode+'--'+case+'.json')
            if path.exists():continue
            result={'status':'running','mode':mode,'case':case,'precision':'original quantized MLX values, FP32 floating parameters/activations','generated_ids':[],'text':''};start=time.time()
            for part in stream_generate(model,processor,f['prompt'],image=[str(f['image'])] if f['image'] else [],max_tokens=64,temperature=0,enable_thinking=False,skip_special_tokens=True):
                if part.token is not None:result['generated_ids'].append(int(part.token))
                result['text']+=part.text;result.update(finish_reason=part.finish_reason,elapsed_seconds=time.time()-start,prompt_tokens=part.prompt_tokens)
                save(root/(mode+'--'+case+'-progress.json'),result)
            result['status']='complete';save(path,result);print(mode,case,result['text'],flush=True)

if __name__=='__main__':main()
