export const MODEL_ID = 'onnx-community/Qwen3.5-0.8B-ONNX';
export const MODEL_REVISION = 'c0d619322dad7c4441a8841a53fc59772ddddcc0';
export const MAX_QUESTION_CHARS = 9000;

const THERMO = /\b(thermodynamics?|enthalpy|internal energy|isothermal|adiabatic|work done|heat engine|first law)\b/i;
const PHYSICS = /\b(force|velocity|acceleration|momentum|electric field|capacitor|current|voltage|lens|mirror|photon|kinetic energy|gas piston)\b/i;
const CHEMISTRY = /\b(mole|reaction|equilibrium|enthalpy|entropy|gibbs|electron configuration|orbital|oxidation|acid|base|chemical|atomisation)\b/i;
const MATH = /\b(integral|derivative|matrix|determinant|polynomial|probability|permutation|triangle|coordinate|function|limit)\b/i;

export function suggestSubject(question, hint = 'auto') {
  if (hint !== 'auto') return { subject: hint, ambiguous: false };
  const p = PHYSICS.test(question);
  const c = CHEMISTRY.test(question);
  const m = MATH.test(question);
  if (p && c) return { subject: 'cross-subject', ambiguous: true };
  if (m && !p && !c) return { subject: 'mathematics', ambiguous: false };
  if (c && !p) return { subject: 'chemistry', ambiguous: false };
  if (p && !c) return { subject: 'physics', ambiguous: false };
  return { subject: 'undetermined', ambiguous: true };
}

export function buildSolverPrompt({ question, hasImage, subjectHint, exam, answerType }) {
  const clean = (question || '').trim().slice(0, MAX_QUESTION_CHARS);
  if (!clean && !hasImage) throw new Error('Add question text or an image.');
  const subject = suggestSubject(clean, subjectHint);
  const context = [
    `Exam: ${exam === 'advanced' ? 'JEE Advanced' : exam === 'main' ? 'JEE Main' : 'JEE, tier unspecified'}.`,
    `Subject: ${subject.subject}${subject.ambiguous ? ' (verify from the problem; state both conventions when genuinely ambiguous)' : ''}.`,
    `Expected answer type: ${answerType === 'multiple' ? 'multiple correct options' : answerType === 'single' ? 'one option' : answerType === 'numeric' ? 'numeric value' : 'infer from question'}.`,
  ];
  if (THERMO.test(clean) || subject.subject === 'cross-subject') {
    context.push('For thermodynamic work, name the sign convention. Chemistry often uses work ON the system w = -P_ext ΔV; physics may use work BY the gas W = +P_ext ΔV. State both if the subject truly cannot be identified.');
  }
  return [
    'Solve the JEE problem carefully. Do not invent missing diagram labels or data. If the image is unreadable, say what is unreadable and stop.',
    ...context,
    'Use a concise, decisive method. Track all requested parts and all possibly correct options. Check units, domains and signs. If uncertain, make the uncertainty explicit.',
    'Return exactly these headings, each on its own line:',
    'GIVEN: key facts and what is asked',
    'PLAN: numbered subgoals and decisive method',
    'WORK: essential equations and calculations',
    'CHECK: one independent consistency check or a specific limitation',
    'FINAL_ANSWER: the option letters, value with units, or a clear unresolved statement',
    'Question:',
    clean || '[The full question is in the attached image.]',
  ].join('\n');
}

export function inspectSolution(raw, answerType = 'infer') {
  const text = String(raw || '').trim();
  const sections = {};
  const pattern = /^(GIVEN|PLAN|WORK|CHECK|FINAL_ANSWER)\s*:\s*([\s\S]*?)(?=^(?:GIVEN|PLAN|WORK|CHECK|FINAL_ANSWER)\s*:|$(?![\s\S]))/gim;
  for (const match of text.matchAll(pattern)) sections[match[1].toUpperCase()] = match[2].trim();
  if (!sections.FINAL_ANSWER) {
    const last = [...text.matchAll(/^FINAL_ANSWER\s*:\s*(.*)$/gim)].at(-1);
    if (last) sections.FINAL_ANSWER = last[1].trim();
  }
  const final = sections.FINAL_ANSWER || '';
  const issues = [];
  if (!final) issues.push('No FINAL_ANSWER line. Generation may have stopped before committing an answer.');
  if (!sections.PLAN) issues.push('No explicit decomposition plan was produced.');
  if (!sections.CHECK) issues.push('No consistency check was produced.');
  if (final && answerType === 'numeric' && !/[+-]?\d/.test(final) && !/unresolved|cannot|unclear/i.test(final)) issues.push('The final line does not appear to contain a number.');
  if (final && answerType === 'multiple' && /^[A-D]$/i.test(final)) issues.push('Only one option was listed for a multiple-correct question; inspect it.');
  return { final, sections, issues, formatComplete: Boolean(final && sections.PLAN && sections.CHECK) };
}
