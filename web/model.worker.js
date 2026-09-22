import { AutoProcessor, Qwen3_5ForConditionalGeneration, RawImage, TextStreamer } from '@huggingface/transformers';
import { MODEL_ID, MODEL_REVISION } from './harness.js';

let processor;
let model;
let loading;
let busy = false;

async function ensureModel() {
  if (model && processor) return;
  if (loading) return loading;
  loading = (async () => {
    if (!self.navigator?.gpu) throw new Error('WebGPU is unavailable. Try a current Chrome or Edge browser on a WebGPU-capable device.');
    self.postMessage({ type: 'status', message: 'Loading tokenizer and image processor…' });
    processor = await AutoProcessor.from_pretrained(MODEL_ID, { revision: MODEL_REVISION });
    self.postMessage({ type: 'status', message: 'Downloading ONNX model into this browser’s cache…' });
    model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
      revision: MODEL_REVISION,
      dtype: { embed_tokens: 'q4', vision_encoder: 'fp16', decoder_model_merged: 'q4' },
      device: 'webgpu',
      progress_callback: (progress) => {
        if (progress.status === 'progress') self.postMessage({ type: 'progress', file: progress.file, loaded: progress.loaded, total: progress.total });
      },
    });
    self.postMessage({ type: 'ready' });
  })();
  try { await loading; } catch (error) { loading = null; model = null; processor = null; throw error; }
}

async function solve({ prompt, imageBuffer, imageType, maxNewTokens }) {
  await ensureModel();
  const content = [];
  let image;
  if (imageBuffer) {
    image = await RawImage.read(new Blob([imageBuffer], { type: imageType }));
    const longest = Math.max(image.width, image.height);
    if (longest > 896) {
      const scale = 896 / longest;
      image = await image.resize(Math.max(1, Math.round(image.width * scale)), Math.max(1, Math.round(image.height * scale)));
    }
    content.push({ type: 'image' });
  }
  content.push({ type: 'text', text: prompt });
  const text = processor.apply_chat_template([{ role: 'user', content }], { add_generation_prompt: true });
  const inputs = image ? await processor(text, image) : await processor(text);
  let streamed = '';
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      streamed += chunk;
      self.postMessage({ type: 'token', text: chunk });
    },
  });
  const outputs = await model.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false, streamer });
  const decoded = processor.batch_decode(outputs.slice(null, [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true })[0] || streamed;
  self.postMessage({ type: 'complete', text: decoded, maxNewTokens });
}

self.onmessage = async ({ data }) => {
  if (busy) { self.postMessage({ type: 'error', message: 'A model operation is already running.' }); return; }
  busy = true;
  try {
    if (data.type === 'load') await ensureModel();
    else if (data.type === 'solve') await solve(data);
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error?.message || error) });
  } finally { busy = false; }
};
