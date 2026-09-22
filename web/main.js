import './style.css';
import { buildSolverPrompt, checkRestAccelerationDisplacement, inspectSolution, MAX_QUESTION_CHARS, suggestSubject } from './harness.js';

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL('./model.worker.js', import.meta.url), { type: 'module' });
let modelReady = false;
let modelLoading = false;
let solving = false;
let selectedFile = null;
let previewUrl = null;
let streamed = '';
let currentAnswerType = 'infer';
let currentQuestion = '';
const fileProgress = new Map();

function state(label, kind = '') {
  $('result-state').textContent = label;
  $('result-state').className = `state-pill ${kind}`;
}
function status(message) { $('model-status').textContent = message; }
function refreshControls() {
  $('load-button').disabled = !navigator.gpu || modelReady || modelLoading || solving;
  $('load-button').textContent = modelReady ? 'Model ready ✓' : modelLoading ? 'Loading model…' : 'Load model locally ↓';
  $('solve-button').disabled = !modelReady || solving;
}
function showError(message) {
  modelLoading = false;
  solving = false;
  state('Needs attention', 'error');
  status(message);
  $('result-empty').hidden = true;
  $('result-content').hidden = false;
  $('raw-output').textContent = streamed || message;
  refreshControls();
}
function showProgress({ file, loaded, total }) {
  if (!file || !total) return;
  fileProgress.set(file, { loaded, total });
  const values = [...fileProgress.values()];
  const amount = values.reduce((n, x) => n + x.loaded, 0);
  const size = values.reduce((n, x) => n + x.total, 0);
  const percent = Math.min(100, Math.round(amount / size * 100));
  $('download-track').hidden = false;
  $('download-bar').style.width = `${percent}%`;
  status(`Downloading ${file.split('/').at(-1)} · ${Math.round(loaded / 1e6)} / ${Math.round(total / 1e6)} MB`);
}

worker.addEventListener('message', ({ data }) => {
  if (data.type === 'status') status(data.message);
  if (data.type === 'progress') showProgress(data);
  if (data.type === 'ready') {
    modelReady = true;
    modelLoading = false;
    $('download-track').hidden = true;
    status('Ready. Inference runs locally through Transformers.js and WebGPU.');
    if (!solving) state('Ready', 'active');
    refreshControls();
  }
  if (data.type === 'token') {
    streamed += data.text;
    $('result-empty').hidden = true;
    $('result-content').hidden = false;
    $('raw-output').textContent = streamed;
    $('raw-output').scrollTop = $('raw-output').scrollHeight;
  }
  if (data.type === 'complete') {
    solving = false;
    const raw = data.text || streamed;
    $('raw-output').textContent = raw;
    const check = inspectSolution(raw, currentAnswerType);
    const calculation = checkRestAccelerationDisplacement(currentQuestion, check.final);
    const issues = [...check.issues, ...(calculation?.issue ? [calculation.issue] : [])];
    $('final-card').hidden = !check.final;
    $('final-answer').textContent = check.final;
    $('format-checks').hidden = !issues.length;
    $('format-checks').replaceChildren();
    if (issues.length) {
      const title = document.createElement('strong');
      title.textContent = 'Review signals (format and limited browser checks):';
      const list = document.createElement('ul');
      for (const issue of issues) { const item = document.createElement('li'); item.textContent = issue; list.append(item); }
      $('format-checks').append(title, list);
    }
    const needsReview = !check.formatComplete || issues.length > 0;
    state(needsReview ? 'Review needed' : 'Finished · unverified', needsReview ? 'error' : 'active');
    status(calculation?.issue ? 'A narrow independent browser calculation disagrees with the model. Inspect the working.' : `Completed. ${check.formatComplete ? 'The requested sections are present.' : 'Some requested sections are missing.'} Most mathematical steps remain unverified.`);
    refreshControls();
  }
  if (data.type === 'error') showError(data.message);
});
worker.addEventListener('error', (error) => showError(`Model worker failed: ${error.message}`));

function setImage(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  $('image-preview').replaceChildren();
  $('image-preview').hidden = true;
  if (!file) return;
  if (!['image/png', 'image/jpeg'].includes(file.type)) { showError('Choose a PNG or JPEG question image.'); $('image').value = ''; return; }
  if (file.size > 10 * 1024 * 1024) { showError('The image exceeds 10 MB. Crop or resize it before retrying.'); $('image').value = ''; return; }
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  const img = document.createElement('img'); img.src = previewUrl; img.alt = 'Attached question';
  const name = document.createElement('span'); name.textContent = file.name;
  const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove'; remove.setAttribute('aria-label', 'Remove image');
  remove.addEventListener('click', () => { $('image').value = ''; setImage(null); });
  $('image-preview').append(img, name, remove);
  $('image-preview').hidden = false;
}

$('image').addEventListener('change', (event) => setImage(event.target.files?.[0] || null));
$('load-button').addEventListener('click', () => {
  if (modelReady || modelLoading) return;
  if (!navigator.gpu) { showError('WebGPU is not available in this browser. This preview needs a WebGPU-capable browser and device.'); return; }
  modelLoading = true;
  fileProgress.clear();
  state('Loading', 'active');
  status('Starting browser model download…');
  refreshControls();
  worker.postMessage({ type: 'load' });
});
$('example-button').addEventListener('click', () => {
  $('question').value = 'A particle starts from rest and moves with a constant acceleration of 2 m/s². Find its displacement in the first 3 s.';
  $('subject').value = 'physics'; $('exam').value = 'main'; $('answer-type').value = 'numeric';
  $('question').focus();
});
$('solve-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!modelReady || solving) return;
  const question = $('question').value.trim();
  if (question.length > MAX_QUESTION_CHARS) return showError(`Keep the question under ${MAX_QUESTION_CHARS} characters.`);
  let prompt;
  try {
    prompt = buildSolverPrompt({ question, hasImage: Boolean(selectedFile), subjectHint: $('subject').value, exam: $('exam').value, answerType: $('answer-type').value });
  } catch (error) { return showError(error.message); }
  currentAnswerType = $('answer-type').value;
  currentQuestion = question;
  const classification = suggestSubject(question, $('subject').value);
  const imageBuffer = selectedFile ? await selectedFile.arrayBuffer() : null;
  streamed = '';
  $('raw-output').textContent = '';
  $('final-card').hidden = true;
  $('format-checks').hidden = true;
  $('result-content').hidden = false;
  $('result-empty').hidden = true;
  solving = true;
  state('Solving', 'active');
  status(`Reading as ${classification.subject}; constructing a plan and answer locally…`);
  refreshControls();
  const request = { type: 'solve', prompt, imageBuffer, imageType: selectedFile?.type, maxNewTokens: Number($('budget').value) };
  worker.postMessage(request, imageBuffer ? [imageBuffer] : []);
});

if (navigator.gpu) status('WebGPU detected. Click “Load model locally” to start the one-time download.');
else { status('WebGPU not detected. Try current Chrome or Edge on a supported device.'); $('load-button').disabled = true; state('WebGPU needed', 'error'); }
refreshControls();
