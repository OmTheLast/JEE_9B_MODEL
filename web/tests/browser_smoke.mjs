import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';

const resultDir = new URL('../.smoke/', import.meta.url);
const imageMode = process.argv.includes('--image');
await mkdir(resultDir, { recursive: true });
const browser = await chromium.launchPersistentContext(new URL('profile/', resultDir).pathname, {
  executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false,
  args: ['--enable-unsafe-webgpu'],
});
const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
const events = [];
page.on('console', message => { if (message.type() === 'error') events.push(`console: ${message.text()}`); });
page.on('pageerror', error => events.push(`page: ${error.message}`));
const save = async (status) => {
  await writeFile(new URL(imageMode ? 'image-result.json' : 'result.json', resultDir), JSON.stringify({ status, mode: imageMode ? 'synthetic image only' : 'synthetic text', time: new Date().toISOString(), events, modelStatus: await page.locator('#model-status').textContent().catch(() => null), resultState: await page.locator('#result-state').textContent().catch(() => null), reviewSignals: await page.locator('#format-checks').textContent().catch(() => null), output: await page.locator('#raw-output').textContent().catch(() => null) }, null, 2));
};
try {
  await page.goto(process.env.JEE_BROWSER_URL || 'http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  await page.screenshot({ path: new URL('page.png', resultDir).pathname, fullPage: true });
  if (!(await page.evaluate(() => Boolean(navigator.gpu)))) throw new Error('WebGPU unavailable in the smoke browser');
  await page.locator('#load-button').click();
  const deadline = Date.now() + 8 * 60 * 1000;
  while (Date.now() < deadline) {
    const label = await page.locator('#result-state').textContent();
    const detail = await page.locator('#model-status').textContent();
    process.stdout.write(`${new Date().toISOString()} ${label}: ${detail}\n`);
    await save('loading');
    if (label?.toUpperCase().includes('READY')) break;
    if (label?.toUpperCase().includes('NEEDS ATTENTION')) throw new Error(detail);
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
  if (!(await page.locator('#result-state').textContent())?.toUpperCase().includes('READY')) throw new Error('Model load timeout');
  if (imageMode) {
    const imageData = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 900; canvas.height = 240;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'black'; ctx.font = '24px Arial';
      ctx.fillText('A particle starts from rest. It accelerates at 2 m/s^2', 25, 85);
      ctx.fillText('for 3 seconds. Find the displacement in metres.', 25, 140);
      return canvas.toDataURL('image/png').split(',')[1];
    });
    await page.locator('#image').setInputFiles({ name: 'synthetic-question.png', mimeType: 'image/png', buffer: Buffer.from(imageData, 'base64') });
    await page.locator('#question').fill('');
  } else {
    await page.locator('#example-button').click();
  }
  await page.locator('#solve-button').click();
  const solveDeadline = Date.now() + 3 * 60 * 1000;
  while (Date.now() < solveDeadline) {
    const label = await page.locator('#result-state').textContent();
    if (label?.toUpperCase().includes('FINISHED') || label?.toUpperCase().includes('REVIEW NEEDED')) break;
    if (label?.toUpperCase().includes('NEEDS ATTENTION')) throw new Error(await page.locator('#model-status').textContent());
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  await save('complete');
  const final = await page.locator('#final-answer').textContent();
  const signals = await page.locator('#format-checks').textContent();
  if (!imageMode && !final) throw new Error('Final answer was not extracted');
  if (!imageMode && !/9/.test(final) && !/independent constant-acceleration check/i.test(signals || '')) throw new Error('A wrong answer was not flagged by the independent check');
  if (imageMode && !(await page.locator('#raw-output').textContent())?.trim()) throw new Error('Image request produced no output');
  process.stdout.write(`FINAL: ${final}; SIGNALS: ${signals?.trim() || 'none'}\n`);
} catch (error) {
  events.push(`smoke: ${error.message}`);
  await save('failed');
  process.exitCode = 1;
} finally { await browser.close(); }
