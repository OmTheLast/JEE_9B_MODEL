import * as ort from 'onnxruntime-web/webgpu';
import { PreTrainedTokenizer } from '@huggingface/transformers';
import { unpackEmbedding, packImage384, makePositions, rotaryAt, repeatingTail } from './original9b-math.js';

export class Original9B {
  constructor(root, notify = () => {}) {
    this.root = root.endsWith('/') ? root : root + '/'; this.notify = notify;
    this.layers = []; this.heads = []; this.embeddings = new Map(); this.adapterCache = new Map(); this.cancelled = false;
  }
  status(message) { this.notify({ type: 'status', message }); }
  async bytes(name) {
    const url = new URL(name, new URL(this.root, self.location.href)).href;
    const expected = this.manifest?.artifacts?.[name]?.bytes;
    let cache; try { cache = await caches.open('llamajee-original9b-v1'); } catch { /* Storage is optional. */ }
    const cached = cache && await cache.match(url);
    if (cached) { const b = await cached.arrayBuffer(); if (!expected || b.byteLength === expected) return b; await cache.delete(url); }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Model file ${name}: HTTP ${response.status}.`);
    const total = expected || Number(response.headers.get('content-length')) || 0;
    const saving = cache ? cache.put(url, response.clone()).catch(() => {}) : Promise.resolve();
    let loaded = 0, previous = 0; const reader = response.body.getReader(); const notify = this.notify;
    const body = new ReadableStream({ async pull(controller) {
      const next = await reader.read(); if (next.done) { controller.close(); return; }
      loaded += next.value.byteLength;
      if (performance.now() - previous > 500) { notify({ type: 'progress', file: name, loaded, total }); previous = performance.now(); }
      controller.enqueue(next.value);
    } });
    const buffer = await new Response(body).arrayBuffer(); await saving;
    if (expected && buffer.byteLength !== expected) throw new Error(`Incomplete download: ${name}. Retry loading.`);
    this.notify({ type: 'progress', file: name, loaded: buffer.byteLength, total: total || buffer.byteLength });
    return buffer;
  }
  async json(name) { return JSON.parse(new TextDecoder().decode(await this.bytes(name))); }
  async load() {
    if (this.layers.length === 32 && this.heads.length === 16) return;
    if (!self.navigator?.gpu) throw new Error('The 9B archive requires WebGPU.');
    this.status('Loading the original 9B model manifest and tokenizer…');
    this.manifest = await this.json('manifest.json');
    this.tokenizer = new PreTrainedTokenizer(await this.json('tokenizer.json'), await this.json('tokenizer_config.json'));
    try {
      for (let i = 0; i < 32; i++) {
        this.status(`Loading 9B decoder block ${i + 1} / 32…`);
        this.layers.push(await ort.InferenceSession.create(await this.bytes(`layer-${String(i).padStart(2, '0')}.onnx`), { executionProviders: ['webgpu'] }));
      }
      for (let i = 0; i < 16; i++) {
        this.status(`Loading 9B output shard ${i + 1} / 16…`);
        this.heads.push(await ort.InferenceSession.create(await this.bytes(`head-${String(i).padStart(2, '0')}.onnx`), { executionProviders: ['webgpu'] }));
      }
    } catch (error) { await this.dispose(); throw error; }
  }
  async adapter(mode) {
    if (this.adapterCache.has(mode)) return this.adapterCache.get(mode);
    if (mode !== 'base' && !this.manifest.adapters[mode]) throw new Error('Unknown archived checkpoint.');
    const zero = Object.fromEntries(Object.entries(this.manifest.adapter_input_shapes).map(([name, dims]) => [name, new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims)]));
    const layers = Array.from({ length: 32 }, () => ({ ...zero }));
    if (mode !== 'base') for (const [i, inputs] of Object.entries(this.manifest.adapters[mode].layers)) {
      for (const [name, meta] of Object.entries(inputs)) layers[Number(i)][name] = new ort.Tensor('float32', new Float32Array(await this.bytes(meta.file)), meta.dims);
    }
    this.adapterCache.set(mode, layers); return layers;
  }
  async embedding(id) {
    const rows = this.manifest.embedding_rows_per_shard, chunk = Math.floor(id / rows), row = id % rows;
    if (!this.embeddings.has(chunk)) {
      const name = `embed-${String(chunk).padStart(2, '0')}`;
      this.embeddings.set(chunk, [new Uint32Array(await this.bytes(name + '.u32')), new Float32Array(await this.bytes(name + '.scales.f32')), new Float32Array(await this.bytes(name + '.biases.f32'))]);
    }
    const [w, s, b] = this.embeddings.get(chunk);
    return unpackEmbedding(w.subarray(row * 512, (row + 1) * 512), s.subarray(row * 64, (row + 1) * 64), b.subarray(row * 64, (row + 1) * 64));
  }
  async prepare(prompt, imageBuffer, imageType) {
    let pixels = null; const content = [];
    if (imageBuffer) {
      const image = await createImageBitmap(new Blob([imageBuffer], { type: imageType }));
      const canvas = new OffscreenCanvas(384, 384), ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 384, 384);
      const scale = Math.min(384 / image.width, 384 / image.height), w = Math.round(image.width * scale), h = Math.round(image.height * scale);
      ctx.drawImage(image, Math.floor((384 - w) / 2), Math.floor((384 - h) / 2), w, h); image.close();
      pixels = packImage384(ctx.getImageData(0, 0, 384, 384).data); content.push({ type: 'image' });
    }
    content.push({ type: 'text', text: prompt });
    let text = this.tokenizer.apply_chat_template([{ role: 'user', content }], { tokenize: false, add_generation_prompt: true, enable_thinking: false });
    if (pixels) text = text.replace('<|image_pad|>', '<|image_pad|>'.repeat(144));
    const ids = this.tokenizer(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number);
    if (ids.length > 1536) throw new Error('This desktop preview supports up to 1,536 input tokens. Shorten the question or crop the image.');
    return { ids, pixels, ...makePositions(ids, pixels ? 248056 : null) };
  }
  async solve({ prompt, imageBuffer, imageType, maxNewTokens = 384, checkpoint = 'base', prepared = null }) {
    await this.load(); this.cancelled = false;
    const input = prepared || await this.prepare(prompt, imageBuffer, imageType), adapters = await this.adapter(checkpoint);
    let features = null, hidden = null; const cfg = this.manifest.config, states = [];
    if (input.pixels) {
      this.status('Loading the optional image encoder (about 1.84 GB)…');
      let vision, tensor, outputs;
      try {
        vision = await ort.InferenceSession.create(await this.bytes('vision.fp32.onnx'), { executionProviders: ['webgpu'] });
        tensor = new ort.Tensor('float32', input.pixels, [576, 1536]); outputs = await vision.run({ pixels: tensor });
        features = new Float32Array(await outputs.image_features.getData());
      } finally { tensor?.dispose(); if (outputs) for (const t of Object.values(outputs)) t.dispose(); if (vision) await vision.release(); }
    }
    const ids = [...input.ids], plen = ids.length, generatedIds = [], start = performance.now(); let imageCursor = 0, text = '', stopReason = 'length', firstTokenScore = null;
    for (const kind of cfg.layer_types) states.push(kind === 'linear_attention' ? [new ort.Tensor('float32', new Float32Array(8192 * 4), [1, 8192, 4]), new ort.Tensor('float32', new Float32Array(32 * 128 * 128), [1, 32, 128, 128])] : [new ort.Tensor('float32', new Float32Array(0), [1, 4, 0, 256]), new ort.Tensor('float32', new Float32Array(0), [1, 4, 0, 256])]);
    try {
      for (let pos = 0; pos < ids.length; pos++) {
        if (this.cancelled) { stopReason = 'cancelled'; break; }
        if (performance.now() - start > 600000) { stopReason = 'time_limit'; break; }
        const vector = features && pos < plen && ids[pos] === 248056 ? features.subarray(imageCursor++ * 4096, imageCursor * 4096) : await this.embedding(ids[pos]);
        hidden = new ort.Tensor('float32', vector, [1, 1, 4096]);
        const position = input.positions.map((p) => pos < plen ? p[pos] : pos + input.delta), rotary = rotaryAt(position);
        const cos = new ort.Tensor('float32', rotary.cos, [1, 1, 64]), sin = new ort.Tensor('float32', rotary.sin, [1, 1, 64]);
        try {
          for (let i = 0; i < 32; i++) {
            const rec = cfg.layer_types[i] === 'linear_attention';
            const feed = { ...adapters[i], hidden, ...(rec ? { conv: states[i][0], state: states[i][1] } : { key: states[i][0], value: states[i][1], cos, sin }) };
            const output = await this.layers[i].run(feed); hidden.dispose(); for (const t of states[i]) t.dispose();
            hidden = output.hidden_out; states[i] = rec ? [output.conv_out, output.state_out] : [output.key_out, output.value_out];
          }
        } finally { cos.dispose(); sin.dispose(); }
        if (pos + 1 >= plen) {
          let best = -Infinity, token = 0;
          for (let i = 0; i < 16; i++) {
            const result = await this.heads[i].run({ hidden }), logits = await result.logits.getData();
            for (let j = 0; j < logits.length; j++) if (logits[j] > best) { best = logits[j]; token = i * this.manifest.head_rows_per_shard + j; }
            result.logits.dispose();
          }
          if (!Number.isFinite(best)) throw new Error('The device returned invalid model scores. Stop and reload the model.');
          if (firstTokenScore === null) firstTokenScore = { id: token, logit: best };
          generatedIds.push(token); text = this.tokenizer.decode(generatedIds, { skip_special_tokens: true });
          this.notify({ type: 'token', fullText: text });
          if ([248044, 248046].includes(token)) { stopReason = 'stop'; break; }
          if (repeatingTail(generatedIds)) { stopReason = 'repetition'; break; }
          if (generatedIds.length >= maxNewTokens) break;
          ids.push(token);
        } else if (pos % 10 === 0) this.status(`Reading the question: ${pos + 1} / ${plen} tokens…`);
        hidden.dispose(); hidden = null;
      }
      return { text, generatedIds, firstTokenScore, inputIds: input.ids, stopReason, seconds: (performance.now() - start) / 1000, maxNewTokens, checkpoint };
    } finally { hidden?.dispose(); for (const pair of states) for (const t of pair) t.dispose(); }
  }
  async dispose() {
    for (const s of [...this.layers, ...this.heads]) await s.release(); this.layers = []; this.heads = [];
    const tensors = new Set(); for (const layers of this.adapterCache.values()) for (const row of layers) for (const t of Object.values(row)) tensors.add(t);
    for (const t of tensors) t.dispose(); this.adapterCache.clear(); this.embeddings.clear();
  }
}
