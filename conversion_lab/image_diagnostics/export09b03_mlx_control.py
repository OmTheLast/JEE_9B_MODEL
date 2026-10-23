"""Original MLX input and generation controls; two synthetic images only."""
import json
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load,prepare_inputs
from run_export09b_mlx_parity import SETUP,attach_adapter,generate_one
from export09b03_controls import OUT,CASES,prepared,save
from export09b_layer_probe import error

def main():
    path=OUT/'mlx-controls.json'
    if path.exists():raise RuntimeError('Preserve existing result')
    mx.set_memory_limit(40*2**30);mx.set_cache_limit(2*2**30);mx.random.seed(0)
    model,processor=load(json.loads((SETUP/'model-download.json').read_text())['path'],trust_remote_code=False);model.freeze();model.eval()
    result={'status':'running','cases':{}}
    for case in CASES:
        dest=OUT/case;_,prompt,_=prepared(case)
        values=prepare_inputs(processor,prompts=[prompt],images=[str(dest/'question.png')],add_special_tokens=False,padding=True,return_tensors='mlx')
        saved=np.load(dest/'processor.npz');comparisons={}
        for key in ['input_ids','pixel_values','image_grid_thw']:
            a=np.array(values[key]);b=saved[key];comparisons[key]={'shape':list(a.shape),'equal':bool(np.array_equal(a,b)),'error':error(a,b)}
        features,_=model.vision_tower(values['pixel_values'].astype(model.vision_tower.patch_embed.proj.weight.dtype),values['image_grid_thw']);mx.eval(features)
        fv=np.array(features.astype(mx.float32));np.save(dest/'vision-original-mlx.npy',fv)
        emb=model.get_input_embeddings(values['input_ids'],values['pixel_values'],image_grid_thw=values['image_grid_thw'],mask=values.get('attention_mask'));mx.eval(emb.position_ids)
        pos=np.array(emb.position_ids);np.save(dest/'positions-original-mlx.npy',pos)
        result['cases'][case]={'processor':comparisons,'positions_equal':bool(np.array_equal(pos,saved['position_ids'])),'position_shape':list(pos.shape),'vision_vs_native_fp32':error(fv,np.load(dest/'vision-native-fp32.npy')),'base':generate_one(model,processor,prompt,dest/'question.png',64)}
        save(OUT/'mlx-controls-progress.json',result)
    attach_adapter(model)
    for case in CASES:
        _,prompt,_=prepared(case);result['cases'][case]['adapted']=generate_one(model,processor,prompt,OUT/case/'question.png',64);save(OUT/'mlx-controls-progress.json',result)
    result['status']='complete';save(path,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
