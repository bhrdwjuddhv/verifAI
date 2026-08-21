/**
 * Runnable check for the live-call risk engine.
 *
 *   node --experimental-strip-types scripts/check-risk.mts
 *
 * The engine decides what a user is shown during a call, and every rule in it (renormalize
 * over present signals, smooth, hold the band until the evidence persists) is the kind of
 * thing that looks right and behaves wrong.
 */

import assert from 'node:assert/strict';
import {
  bandFor,
  fuseWindow,
  initialState,
  parseCallWeights,
  pushWindow,
  timbreDistance,
  type RiskState,
} from '../lib/call/risk.ts';

// --- fusion ------------------------------------------------------------------------------
assert.equal(fuseWindow({ audio: 80, video: 20 }).risk, 50);
assert.equal(fuseWindow({ audio: 90, video: null }).risk, 90, 'a missing signal must not dilute');
assert.equal(fuseWindow({ audio: null, video: 40 }).risk, 40);
assert.equal(fuseWindow({ audio: null, video: null }).risk, null);
assert.deepEqual(fuseWindow({ audio: 90, video: null }).used, ['audio']);
assert.equal(fuseWindow({ audio: 80, video: 20 }, { audio: 3, video: 1 }).risk, 65, 'weights are relative');

// --- weight parsing ----------------------------------------------------------------------
assert.deepEqual(parseCallWeights('audio=0.7,video=0.3'), { audio: 0.7, video: 0.3 });
assert.deepEqual(parseCallWeights(''), { audio: 0.5, video: 0.5 });
assert.deepEqual(parseCallWeights('bogus=1'), { audio: 0.5, video: 0.5 }, 'unknown keys ignored, not fatal');
assert.deepEqual(parseCallWeights('audio=0,video=0'), { audio: 0.5, video: 0.5 }, 'all-zero falls back');

// --- bands --------------------------------------------------------------------------------
assert.equal(bandFor(5), 'low');
assert.equal(bandFor(25), 'uncertain');
assert.equal(bandFor(50), 'suspicious');
assert.equal(bandFor(95), 'high');

// --- smoothing + hysteresis ---------------------------------------------------------------
const opts = { alpha: 0.4, hysteresis: 3 };

// A quiet call: silence changes nothing at all.
let s: RiskState = initialState();
s = pushWindow(s, { audio: null, video: null }, opts);
assert.equal(s.band, 'idle');
assert.equal(s.skipped, 1);
assert.equal(s.scored, 0);
assert.equal(s.smoothed, null, 'silence must not be read as safety');

// First scored window adopts its band immediately — no three-window "Listening…" limbo.
s = pushWindow(s, { audio: 4, video: null }, opts);
assert.equal(s.band, 'low');
assert.equal(s.trust, 96);

// One bad window must NOT move the displayed band.
const before = s.band;
s = pushWindow(s, { audio: 100, video: null }, opts);
assert.equal(s.band, before, 'a single spike must not flip the band');
assert.ok(s.pendingCount >= 1);

// Sustained synthetic evidence must get there — and smoothly.
const trail: number[] = [];
for (let i = 0; i < 10; i++) {
  s = pushWindow(s, { audio: 95, video: 90 }, opts);
  trail.push(s.smoothed!);
}
assert.equal(s.band, 'high', `sustained synthetic audio must reach high, got ${s.band}`);
assert.ok(trail[0] < trail[trail.length - 1], 'smoothed risk must rise, not jump');
for (let i = 1; i < trail.length; i++) {
  assert.ok(trail[i] >= trail[i - 1], 'EMA must be monotonic under a constant input');
}
assert.ok(s.consecutiveHigh >= 3, 'the alert counter tracks consecutive high windows');

// Recovery is equally deliberate: back to real, band holds until the evidence persists.
const highState = { ...s };
let r = pushWindow(highState, { audio: 2, video: 2 }, opts);
assert.equal(r.band, 'high', 'one clean window must not clear a high band');
assert.equal(r.consecutiveHigh, 0, 'but the alert counter resets at once');
for (let i = 0; i < 12; i++) r = pushWindow(r, { audio: 2, video: 2 }, opts);
assert.ok(r.band === 'low' || r.band === 'uncertain', `should recover, got ${r.band}`);

// Skipped windows during a high band hold it — they are not evidence of safety.
const held = pushWindow({ ...s }, { audio: null, video: null }, opts);
assert.equal(held.band, 'high');
assert.equal(held.smoothed, s.smoothed);

// --- timbre drift ------------------------------------------------------------------------
assert.equal(timbreDistance([1, 2, 3], [1, 2, 3]), 0, 'identical timbre is zero distance');
assert.ok((timbreDistance([1, 2, 3], [3, 2, 1]) ?? 0) > 0.1, 'a different spectral shape must register');
assert.equal(timbreDistance([1, 2, 3], [2, 4, 6]), 0, 'louder but identical shape is not drift');
assert.equal(timbreDistance([], [1]), null);
assert.equal(timbreDistance([0, 0], [1, 1]), null, 'a silent reference cannot be compared');

console.log('risk engine selfcheck passed');
