import { Original9B } from './original9b.js';
const root = import.meta.env.VITE_ARCHIVE_MODEL_ROOT;
const evidence = import.meta.env.VITE_DIAGNOSTIC_ROOT;
const control = import.meta.env.VITE_CONTROL_ROOT;
const status = document.querySelector('#status'), output = document.querySelector('#result');
document.querySelector('#run').onclick = async () => {
  document.querySelector('#run').disabled = true;
  const model = new Original9B(root, (message) => { if (message.message) status.textContent = message.message; else if (message.type === 'token') status.textContent = message.fullText; });
  const result = { status: 'running', checks: [], pixelChecks: [], scope: 'Synthetic fixture parity; not JEE accuracy', started: new Date().toISOString() };
  const show = () => { output.textContent = JSON.stringify(result, null, 2); };
  try {
    await model.load();
    let cases = [['text', 'base'], ...Object.keys(model.manifest.adapters).map((mode) => ['text', mode]), ['addition', 'base'], ['subtraction', 'base'], ['addition', 'reliable01-step060'], ['subtraction', 'reliable01-step060']];
    const activation = new URLSearchParams(location.search).has('activation');
    if (activation) cases = cases.filter(([name]) => name === 'text');
    const prepared = new Map();
    for (const [caseName, checkpoint] of cases) {
      if (!prepared.has(caseName)) {
        const image = caseName !== 'text';
        const fixture = await (await fetch(image ? `${evidence}/${caseName}/fixture.json` : `${control}/text-fixture.json`)).json();
        const prompt = image ? 'Read the arithmetic in the image. Reply in one line: FINAL_ANSWER: followed by the result.' : 'What is 2 + 3? Reply in one line: FINAL_ANSWER: followed by the result.';
        const buffer = image ? await (await fetch(`${evidence}/${caseName}/question.png`)).arrayBuffer() : null;
        const input = await model.prepare(prompt, buffer, 'image/png');
        const check = { case: caseName, tokenIdentity: JSON.stringify(input.ids) === JSON.stringify(fixture.input_ids) };
        if (image) {
          const pixels = new Float32Array(await (await fetch(`${evidence}/${caseName}/pixels.f32`)).arrayBuffer()); let max = 0;
          for (let i = 0; i < pixels.length; i++) max = Math.max(max, Math.abs(pixels[i] - input.pixels[i]));
          check.maxPixelError = max; check.positionIdentity = JSON.stringify(input.positions) === JSON.stringify(fixture.position_ids); check.deltaIdentity = input.delta === fixture.rope_delta;
        }
        result.pixelChecks.push(check); show();
        if (!check.tokenIdentity || (image && (check.maxPixelError > 1e-6 || !check.positionIdentity || !check.deltaIdentity))) throw new Error(`Prepared input mismatch: ${caseName}`);
        prepared.set(caseName, input);
      }
      status.textContent = `Running ${checkpoint} / ${caseName}…`;
      const answer = await model.solve({ prepared: prepared.get(caseName), checkpoint, maxNewTokens: 64 });
      const expected = await (await fetch(`${control}/mlx-controls/${checkpoint}--${caseName}.json`)).json();
      const native = await (await fetch(`${control}/onnx-controls/${checkpoint}--${caseName}.json`)).json();
      const logitError = Math.abs(answer.firstTokenScore.logit - native.first_top10[0].logit);
      result.checks.push({ logitError, scoreMatches: answer.firstTokenScore.id === native.first_top10[0].id && logitError < 0.001, case: caseName, checkpoint, exactTokenMatch: JSON.stringify(answer.generatedIds) === JSON.stringify(expected.generated_ids), ...answer }); show();
    }
    result.status = 'complete'; result.passed = result.checks.every((c) => c.exactTokenMatch && c.scoreMatches);
  } catch (error) { result.status = 'failed'; result.error = String(error); }
  finally { await model.dispose(); result.resourcesReleased = true; show(); status.textContent = result.status; }
};
