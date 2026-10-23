"""Create an answer-free prompt with a synthetic arithmetic image for export QA."""
import json
from types import SimpleNamespace,MethodType
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from transformers import AutoConfig,AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
import onnxruntime as ort
from export09b_layer_probe import SOURCE,OUT
from checkpoint import atomic_write


def main():
    out=OUT/'image-arithmetic-v1';out.mkdir(exist_ok=True)
    if (out/'fixture.json').exists():raise RuntimeError('Fixture exists; preserve it')
    image=Image.new('RGB',(384,384),'white');draw=ImageDraw.Draw(image)
    draw.text((38,155),'2 + 3 = ?',font=ImageFont.load_default(size=56),fill='black');image.save(out/'question.png')
    processor=AutoProcessor.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True)
    prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':'Read the arithmetic in the image. Reply in one line: FINAL_ANSWER: followed by the result.'}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    values=processor(text=[prompt],images=[image],do_resize=False,return_tensors='pt')
    cfg=AutoConfig.from_pretrained(SOURCE/'upstream-bf16',local_files_only=True)
    owner=SimpleNamespace(config=cfg)
    owner.get_vision_position_ids=MethodType(Qwen3_5Model.get_vision_position_ids,owner)
    types=values.get('mm_token_type_ids',(values['input_ids']==cfg.image_token_id).int())
    positions,delta=Qwen3_5Model.get_rope_index(owner,values['input_ids'],types,image_grid_thw=values['image_grid_thw'])
    pixels=values['pixel_values'].numpy().astype(np.float32);pixels.tofile(out/'pixels.f32')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(OUT/'vision384-v1/vision.fp32.onnx'),options,providers=['CPUExecutionProvider'])
    features=session.run(None,{'pixels':pixels})[0];features.tofile(out/'features.f32')
    ids=values['input_ids'][0].tolist();expanded=processor.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    assert processor.tokenizer(expanded,add_special_tokens=False).input_ids==ids
    fixture={'prompt':expanded,'input_ids':ids,'position_ids':positions[:,0,:].tolist(),'rope_delta':int(delta[0,0]),
             'image_token_id':cfg.image_token_id,'pixel_shape':list(pixels.shape),'feature_shape':list(features.shape),'max_new_tokens':64,
             'scope':'synthetic image arithmetic; no answer in prompt; not a JEE evaluation'}
    atomic_write(out/'fixture.json',json.dumps(fixture,indent=2)+'\n');print(len(ids),features.shape)


if __name__=='__main__':main()
