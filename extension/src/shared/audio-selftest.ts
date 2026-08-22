/**
 * The self-test vector, and the rule for trusting it.
 *
 * `audio_selftest.json` stores only the first 1600 of its 48000 samples, while `expected_prob`
 * was computed on all 48000 — so feeding the stored prefix into the chain cannot reproduce the
 * expected number, and a gate built on that would be measuring nothing.
 *
 * The vector is a documented deterministic three-tone signal, so it is regenerated here and
 * then checked two ways before use: the stored prefix must match sample-for-sample, and the
 * sum of absolute values must match `full_audio_checksum`. If either fails this returns null
 * and on-device audio stays off — a self-test run on the wrong input proves nothing.
 *
 * This mirrors `reconstruct_selftest_audio` in scripts/common/audio_v2.py exactly. The two are
 * kept in step by scripts/check-liveguard.mjs, which asserts this reproduces the checksum in
 * the shipped JSON.
 */

export interface AudioSelftestMeta {
  sample_rate: number;
  duration_sec: number;
  num_samples: number;
  audio: number[];
  full_audio_checksum?: number;
  expected_prob: number;
  tol: number;
  pipeline?: string;
}

export const SELFTEST_SR = 16000;

export interface Reconstructed {
  samples: Float32Array | null;
  reason: string | null;
}

export function reconstructSelftestAudio(meta: AudioSelftestMeta): Reconstructed {
  const stored = meta?.audio ?? [];
  const total = Number(meta?.num_samples ?? 0);
  if (stored.length === 0 || total <= 0) {
    return { samples: null, reason: 'self-test JSON carries no audio vector' };
  }

  if (stored.length === total) {
    return { samples: Float32Array.from(stored), reason: null };
  }

  const rate = Number(meta.sample_rate ?? SELFTEST_SR);
  if (rate !== SELFTEST_SR) {
    return { samples: null, reason: `self-test sample rate ${rate} is not ${SELFTEST_SR}` };
  }

  const seconds = Number(meta.duration_sec ?? total / SELFTEST_SR);
  const rebuilt = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    // np.linspace(0, seconds, total, endpoint=False)
    const t = (seconds * i) / total;
    rebuilt[i] =
      0.6 * Math.sin(2 * Math.PI * 440 * t) +
      0.3 * Math.sin(2 * Math.PI * 880 * t) +
      0.1 * Math.sin(2 * Math.PI * 1320 * t);
  }

  for (let i = 0; i < stored.length; i++) {
    if (Math.abs(rebuilt[i] - stored[i]) > 1e-5) {
      return {
        samples: null,
        reason:
          'self-test vector is truncated and the reconstruction does not match its stored ' +
          'prefix — regenerate the JSON with the full vector',
      };
    }
  }

  const checksum = meta.full_audio_checksum;
  if (checksum !== undefined && checksum !== null) {
    let total_abs = 0;
    for (let i = 0; i < total; i++) total_abs += Math.abs(rebuilt[i]);
    if (Math.abs(total_abs - Number(checksum)) > 1e-2) {
      return {
        samples: null,
        reason:
          'self-test vector is truncated and the reconstruction does not match ' +
          'full_audio_checksum — regenerate the JSON with the full vector',
      };
    }
  }

  return { samples: rebuilt, reason: null };
}
