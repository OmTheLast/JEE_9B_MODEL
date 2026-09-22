import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSolverPrompt, checkRestAccelerationDisplacement, inspectSolution, suggestSubject } from '../harness.js';

test('a text or image is required, and screenshot-only input is represented', () => {
  assert.throws(() => buildSolverPrompt({ question: '', hasImage: false, subjectHint: 'auto', exam: 'main', answerType: 'infer' }));
  const prompt = buildSolverPrompt({ question: '', hasImage: true, subjectHint: 'auto', exam: 'advanced', answerType: 'multiple' });
  assert.match(prompt, /full question is in the attached image/);
  assert.match(prompt, /all possibly correct options/);
});

test('narrow independent kinematics check catches a plausible wrong formula', () => {
  const question = 'A particle starts from rest and moves with a constant acceleration of 2 m/s². Find its displacement in the first 3 s.';
  assert.deepEqual(checkRestAccelerationDisplacement(question, '9 m')?.agrees, true);
  const mismatch = checkRestAccelerationDisplacement(question, '6 m');
  assert.equal(mismatch?.expected, 9);
  assert.equal(mismatch?.observed, 6);
  assert.match(mismatch?.issue, /displacement formula/);
  assert.equal(checkRestAccelerationDisplacement('Find the velocity after 3 s.', '6 m/s'), null);
});

test('thermodynamics sign conventions and subject hints remain explicit', () => {
  const prompt = buildSolverPrompt({ question: 'A gas expands isothermally. Find work done.', hasImage: false, subjectHint: 'chemistry', exam: 'main', answerType: 'numeric' });
  assert.match(prompt, /work ON the system/);
  assert.match(prompt, /work BY the gas/);
  assert.match(prompt, /Subject: chemistry/);
  assert.equal(suggestSubject('Find the derivative of a polynomial').subject, 'mathematics');
});

test('format inspection does not mistake a parseable answer for verified science', () => {
  const good = inspectSolution('GIVEN: data\nPLAN: use a formula\nWORK: wrong math\nCHECK: claimed check\nFINAL_ANSWER: 7', 'numeric');
  assert.equal(good.formatComplete, true);
  assert.equal(good.final, '7');
  assert.deepEqual(good.issues, []);
  const incomplete = inspectSolution('PLAN: try a method\nWORK: unfinished', 'numeric');
  assert.equal(incomplete.formatComplete, false);
  assert.ok(incomplete.issues.some(x => x.includes('FINAL_ANSWER')));
  const markdown = inspectSolution('**GIVEN:** values\n**PLAN:** one step\n**WORK:** 3²\n**CHECK:** units\n**FINAL_ANSWER:**\nA: 9', 'numeric');
  assert.equal(markdown.final, 'A: 9');
  assert.equal(markdown.formatComplete, true);
  assert.ok(markdown.issues.some(x => x.includes('option-style prefix')));
  const imageStyle = inspectSolution('### **GIVEN**\nrest\n### **PLAN**\nuse s = ½at²\n### **WORK**\ns = 9\n### **CHECK**\nunits m\n### **FINAL_ANSWER**\n**9**', 'numeric');
  assert.equal(imageStyle.final, '9');
  assert.equal(imageStyle.formatComplete, true);
});
