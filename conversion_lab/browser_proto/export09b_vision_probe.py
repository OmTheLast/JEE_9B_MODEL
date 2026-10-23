"""Fixed 384x384 image export check for the actual pinned 9B vision encoder.

This probe does not establish dynamic-resolution support or image-answer quality.
"""
import gc
import json
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig,AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from onnxruntime.transformers.float16 import convert_float_to_float16
from export09b_layer_probe import SOURCE,OUT,error
from export09b_fixtures import ensure_synthetic_image,sha256
from checkpoint import atomic_write


class Vision(torch.nn.Module):
    def __init__(self,model,grid):
        super().__init__();self.model=model;self.register_buffer('grid',grid)
    def forward(self,pixels):return self.model(pixels,self.grid).pooler_output


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    out=OUT/'vision384-v1';out.mkdir(exist_ok=True)
    if (out/'progress.json').exists():raise RuntimeError('Preserve earlier probe; use new version')
    result={'status':'loading','scope':'fixed384 image encoder only; not end-to-end image solving','started':time.time()}
    def save():atomic_write(out/'progress.json',json.dumps(result,indent=2)+'\n')
    save()
    cfg=AutoConfig.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True).vision_config;cfg._attn_implementation='eager'
    with torch.device('meta'):model=Qwen3_5VisionModel(cfg)
    index=json.loads((SOURCE/'upstream-bf16/model.safetensors.index.json').read_text())['weight_map'];weights={}
    for name,shard in index.items():
        if name.startswith('model.visual.'):
            with safe_open(SOURCE/'upstream-bf16'/shard,framework='pt') as f:weights[name[len('model.visual.'):]]=f.get_tensor(name).float()
    model.load_state_dict(weights,strict=True,assign=True);model.eval()
    # Nonpersistent rotary buffers originate on meta; materialize via constructor.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
    model.rotary_pos_emb=Qwen3_5VisionRotaryEmbedding(cfg)
    processor=AutoProcessor.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True)
    image=Image.open(ensure_synthetic_image()).convert('RGB').resize((384,384));image.save(out/'synthetic384.png')
    values=processor.image_processor(images=[image],do_resize=False,return_tensors='pt')
    pixels=values['pixel_values'].float();grid=values['image_grid_thw']
    wrapper=Vision(model,grid).eval();expected=wrapper(pixels).numpy().copy()
    result.update(status='exporting',pixel_shape=list(pixels.shape),grid=grid.tolist(),output_shape=list(expected.shape));save()
    path=out/'vision.fp32.onnx'
    torch.onnx.export(wrapper,(pixels,),str(path),input_names=['pixels'],output_names=['image_features'],opset_version=18,dynamo=False,do_constant_folding=True)
    pixels.numpy().tofile(out/'pixels.f32');expected.tofile(out/'reference.f32')
    del wrapper,model,weights;gc.collect()
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    values=session.run(None,{'pixels':pixels.numpy()})[0];result['onnx_fp32_error']=error(values,expected)
    assert np.allclose(values,expected,atol=2e-3,rtol=2e-3),result['onnx_fp32_error']
    del session;gc.collect();result['status']='converting-fp16';save()
    graph=convert_float_to_float16(onnx.load(path),keep_io_types=True,disable_shape_infer=True)
    half=out/'vision.fp16.onnx';onnx.save(graph,half);del graph;gc.collect()
    session=ort.InferenceSession(str(half),options,providers=['CPUExecutionProvider']);actual=session.run(None,{'pixels':pixels.numpy()})[0]
    result['fp16_error']=error(actual,expected);actual.tofile(out/'fp16-cpu.f32')
    result.update(status='native-complete-browser-pending',elapsed_seconds=time.time()-result['started'],files={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in [path,half]})
    save();atomic_write(out/'result.json',json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
