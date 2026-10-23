// Replace the release revision only after verified publication.
export const ARCHIVE_ROOT = import.meta.env.VITE_ARCHIVE_MODEL_ROOT || 'https://huggingface.co/OmTheLast/LlamaJEE-9B-ONNX-WebGPU/resolve/81f9f6a2591a348994e655a8ea8302c392b9dce2/';
export const ARCHIVED_MODELS = {
  base: '9B · unchanged base',
  'coverage16-step060': '9B · 30 families · 16 layers · step 60',
  'data60-step060': '9B · 60 families · 3 layers · step 60',
  'reliable01-step030': '9B · 30 families · 3 layers · step 30',
  'reliable01-step060': '9B · 30 families · 3 layers · step 60',
  'reliable01-step090': '9B · 30 families · 3 layers · step 90',
};
