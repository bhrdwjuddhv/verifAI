// ──────────────────────────────────────────────────────────────
// lib/eval/sampleGenerator.ts — Deterministic 300-sample eval dataset
// ──────────────────────────────────────────────────────────────
//
// Generates 300 synthetic evaluation samples with realistic filenames,
// module scores, and ground-truth labels. The dataset is deterministic
// (seeded by index) so samples are stable across sessions.
//
// Distribution: ~40% authentic, ~40% manipulated, ~20% inconclusive/suspicious
// Split: first 240 → working set, last 60 → held-out set

import type { EvalSample, ModuleScores, FusionWeights } from './types';
import { DEFAULT_FUSION_WEIGHTS } from './types';
import type { VerdictCategory } from '@/lib/verdict';

// ── Deterministic pseudo-random number generator (mulberry32) ──
function mulberry32(seed: number) {
  return function (): number {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Realistic filename pools ──
const AUTHENTIC_FILENAMES = [
  'IMG_20240315_142356.jpg', 'DSC_0847.NEF', 'PXL_20240801_091523.jpg',
  'DCIM_photo_0412.jpg', 'Canon_EOS_R5_raw_001.CR3', 'shot_portrait_natural.jpg',
  'family_dinner_real.png', 'landscape_sunset_original.jpg', 'selfie_morning.heic',
  'wedding_ceremony_raw.jpg', 'baby_first_steps.mov', 'concert_live_audio.wav',
  'interview_recording_original.mp3', 'news_broadcast_unedited.mp4',
  'security_cam_feed_0023.avi', 'drone_aerial_coastline.jpg',
  'gopro_surfing_clip.mp4', 'iphone15_nightmode.heic', 'samsung_galaxy_hdr.jpg',
  'pixel_8_astrophoto.jpg', 'nikon_d850_studio.jpg', 'sony_a7iv_portrait.arw',
  'street_photography_001.jpg', 'nature_wildlife_eagle.jpg',
];

const FAKE_FILENAMES = [
  'deepfacelab_swap_final.png', 'roop_faceswap_output.jpg', 'simswap_result_v2.png',
  'midjourney_v6_fantasy_castle.png', 'dalle3_photorealistic_person.jpg',
  'stable_diffusion_xl_portrait.png', 'flux1_generated_landscape.jpg',
  'comfyui_workflow_output.png', 'firefly_synthetic_scene.jpg',
  'sora_video_dreamscape.mp4', 'runway_gen3_commercial.mp4',
  'pika_animation_output.mp4', 'luma_dream_machine_clip.mp4',
  'wav2lip_dubbed_speech.mp4', 'sadtalker_puppet_interview.mp4',
  'liveportrait_reenactment.mp4', 'elevenlabs_voice_clone.mp3',
  'rvc_converted_voice.wav', 'bark_tts_narration.mp3',
  'fake_face_celebrity_swap.jpg', 'ai_generated_headshot.png',
  'synthetic_news_anchor.mp4', 'generated_product_photo.jpg',
  'deepfake_political_speech.mp4',
];

const EDITED_FILENAMES = [
  'faceapp_smoothed_selfie.jpg', 'remini_enhanced_old_photo.jpg',
  'photoshop_retouched_model.psd', 'lightroom_color_graded.jpg',
  'canva_template_modified.png', 'snapseed_filtered_landscape.jpg',
  'facetune_beauty_filter.jpg', 'instasize_cropped_post.jpg',
  'vsco_film_preset_applied.jpg', 'afterlight_vintage_effect.jpg',
  'suspicious_compression_artifact.jpg', 'modified_exif_stripped.jpg',
];

/** Compute a weighted fusion score from module scores and weights. */
export function computeWeightedScore(modules: ModuleScores, weights: FusionWeights): number {
  const totalWeight =
    weights.faceForgery + weights.frequencyGan + weights.audioSpoof + weights.exifEla + weights.c2pa;
  if (totalWeight === 0) return 50;

  const raw =
    (modules.faceForgery * weights.faceForgery +
      modules.frequencyGan * weights.frequencyGan +
      modules.audioSpoof * weights.audioSpoof +
      modules.exifEla * weights.exifEla +
      modules.c2pa * weights.c2pa) /
    totalWeight;

  return Math.round(Math.min(100, Math.max(0, raw)));
}

/** Map a numeric score to a verdict category. Same thresholds as lib/verdict.ts getVerdict(). */
export function scoreToVerdict(score: number): VerdictCategory {
  if (score >= 75) return 'genuine';
  if (score >= 40) return 'suspicious';
  return 'manipulated';
}

/**
 * Generate all 300 evaluation samples.
 * @param weights — fusion weights to compute predicted scores (defaults to equal)
 */
export function generateEvalSamples(weights: FusionWeights = DEFAULT_FUSION_WEIGHTS): EvalSample[] {
  const rng = mulberry32(42); // fixed seed for determinism
  const samples: EvalSample[] = [];

  for (let i = 0; i < 300; i++) {
    const r = rng();
    let groundTruthLabel: EvalSample['groundTruthLabel'];
    let filename: string;
    let moduleScores: ModuleScores;

    if (r < 0.40) {
      // ── Authentic sample ──
      groundTruthLabel = 'authentic';
      filename = AUTHENTIC_FILENAMES[Math.floor(rng() * AUTHENTIC_FILENAMES.length)];
      // High module scores with some natural variance
      moduleScores = {
        faceForgery: 85 + Math.floor(rng() * 15),
        frequencyGan: 82 + Math.floor(rng() * 18),
        audioSpoof: 88 + Math.floor(rng() * 12),
        exifEla: 80 + Math.floor(rng() * 20),
        c2pa: 75 + Math.floor(rng() * 25),
      };
    } else if (r < 0.80) {
      // ── Manipulated sample ──
      groundTruthLabel = 'manipulated';
      filename = FAKE_FILENAMES[Math.floor(rng() * FAKE_FILENAMES.length)];
      // Low module scores
      moduleScores = {
        faceForgery: 5 + Math.floor(rng() * 30),
        frequencyGan: 8 + Math.floor(rng() * 28),
        audioSpoof: 10 + Math.floor(rng() * 25),
        exifEla: 5 + Math.floor(rng() * 35),
        c2pa: 0 + Math.floor(rng() * 20),
      };
    } else {
      // ── Inconclusive / borderline sample ──
      groundTruthLabel = 'inconclusive';
      filename = EDITED_FILENAMES[Math.floor(rng() * EDITED_FILENAMES.length)];
      // Mid-range scores
      moduleScores = {
        faceForgery: 40 + Math.floor(rng() * 30),
        frequencyGan: 35 + Math.floor(rng() * 35),
        audioSpoof: 50 + Math.floor(rng() * 30),
        exifEla: 30 + Math.floor(rng() * 40),
        c2pa: 25 + Math.floor(rng() * 45),
      };
    }

    // Add a unique suffix to filename to prevent duplicates
    const ext = filename.substring(filename.lastIndexOf('.'));
    const base = filename.substring(0, filename.lastIndexOf('.'));
    const uniqueFilename = `${base}_${String(i + 1).padStart(3, '0')}${ext}`;

    const predictedScore = computeWeightedScore(moduleScores, weights);
    const predictedVerdict = scoreToVerdict(predictedScore);

    // Ground truth score: authentic→85-100, manipulated→0-25, inconclusive→35-65
    let groundTruthScore: number | undefined;
    if (groundTruthLabel === 'authentic') {
      groundTruthScore = 85 + Math.floor(rng() * 16);
    } else if (groundTruthLabel === 'manipulated') {
      groundTruthScore = Math.floor(rng() * 26);
    } else {
      groundTruthScore = 35 + Math.floor(rng() * 31);
    }

    samples.push({
      id: `EVAL-${String(i + 1).padStart(3, '0')}`,
      mediaRef: uniqueFilename,
      groundTruthLabel,
      groundTruthScore,
      predictedScore,
      predictedVerdict,
      moduleScores,
      split: i < 240 ? 'working' : 'heldout',
    });
  }

  return samples;
}
