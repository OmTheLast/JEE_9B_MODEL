import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSolverPrompt, inspectSolution, suggestSubject } from '../harness.js';

test('a text or image is required, and screenshot-only input is represented', () => {
  assert.throws(() => buildSolverPrompt({ question: '', hasImage: false, subjectHint: 'auto', exam: 'main', answerType: 'infer' }));
  const prompt = buildSolverPrompt({ question: '', hasImage: true, subjectHint: 'auto', exam: 'advanced', answerType: 'multiple' });
  assert.match(prompt, /full question is in the attached image/);
  assert.match(prompt, /all possibly correct options/);
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
});
