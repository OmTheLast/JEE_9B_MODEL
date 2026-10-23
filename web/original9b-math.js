// Packing and position helpers for the pinned, fixed-384 Qwen image encoder.
export function unpackEmbedding(packed, scales, biases) {
  const out = new Float32Array(4096);
  for (let j = 0; j < 4096; j++) out[j] = ((packed[j >>> 3] >>> ((j & 7) * 4)) & 15) * scales[j >>> 6] + biases[j >>> 6];
  return out;
}

export function packImage384(rgba) {
  if (rgba.length !== 384 * 384 * 4) throw new Error('Expected a 384 × 384 RGBA image.');
  const out = new Float32Array(576 * 1536); let at = 0;
  for (let gy = 0; gy < 12; gy++) for (let gx = 0; gx < 12; gx++)
    for (let my = 0; my < 2; my++) for (let mx = 0; mx < 2; mx++)
      for (let c = 0; c < 3; c++) for (let t = 0; t < 2; t++)
        for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
          const pixel = (((gy * 2 + my) * 16 + y) * 384 + (gx * 2 + mx) * 16 + x) * 4 + c;
          out[at++] = Math.fround((Math.fround(rgba[pixel] / 255) - 0.5) / 0.5);
        }
  return out;
}

export function makePositions(ids, imageId) {
  const start = imageId == null ? -1 : ids.indexOf(imageId);
  if (start < 0) return { positions: [0, 1, 2].map(() => ids.map((_, i) => i)), delta: 0 };
  if (ids.filter((id) => id === imageId).length !== 144 || ids.slice(start, start + 144).some((id) => id !== imageId)) throw new Error('Image token layout does not match the fixed encoder.');
  const positions = [[], [], []];
  for (let i = 0; i < ids.length; i++) {
    if (i < start) for (const p of positions) p.push(i);
    else if (i < start + 144) { const j = i - start; positions[0].push(start); positions[1].push(start + Math.floor(j / 12)); positions[2].push(start + j % 12); }
    else for (const p of positions) p.push(i - 132);
  }
  return { positions, delta: -132 };
}

export function rotaryAt(position) {
  const cos = new Float32Array(64), sin = new Float32Array(64);
  for (let j = 0; j < 32; j++) {
    const axis = j % 3 === 1 ? 1 : j % 3 === 2 && j < 30 ? 2 : 0;
    const angle = position[axis] / 1e7 ** (j / 32);
    cos[j] = cos[j + 32] = Math.cos(angle); sin[j] = sin[j + 32] = Math.sin(angle);
  }
  return { cos, sin };
}

export function repeatingTail(ids) {
  for (let n = 8; n <= 64 && n * 3 <= ids.length; n++) {
    let same = true;
    for (let j = 0; j < n * 2; j++) if (ids[ids.length - 1 - j] !== ids[ids.length - 1 - j - n]) { same = false; break; }
    if (same) return true;
  }
  return false;
}
