/**
 * Browser-side capture helpers for a live call: WAV encoding and frame grabs.
 *
 * WAV rather than WebM/Opus because the model service decodes WAV and PCM only — putting
 * ffmpeg in the runtime image to accept Opus would cost more than encoding 3 seconds of audio
 * here does.
 */

/** Float32 mono samples -> a 16-bit PCM WAV blob. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Collects fixed-length windows of the remote audio track.
 *
 * The remote MediaStream is deliberately NOT connected to the destination — this taps it for
 * analysis only, and never plays it twice.
 */
export class AudioWindower {
  private ctx: AudioContext;
  private processor: ScriptProcessorNode;
  private source: MediaStreamAudioSourceNode;
  private chunks: Float32Array[] = [];
  private collected = 0;
  private readonly target: number;

  constructor(stream: MediaStream, seconds: number, private onWindow: (wav: Blob, rate: number) => void) {
    this.ctx = new AudioContext({ sampleRate: 16000 });
    this.source = this.ctx.createMediaStreamSource(stream);
    // ScriptProcessor is deprecated in favour of AudioWorklet, which needs a separate module
    // file and a build step to serve it. For 3-second windows the cost is irrelevant.
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1);
    this.target = Math.round(this.ctx.sampleRate * seconds);

    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
      this.collected += input.length;
      if (this.collected >= this.target) this.flush();
    };

    this.source.connect(this.processor);
    // A ScriptProcessor only runs while connected to the graph. Routing it through a silent
    // gain node keeps it running without duplicating the caller's voice into the speakers.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.processor.connect(mute);
    mute.connect(this.ctx.destination);
  }

  private flush() {
    const merged = new Float32Array(this.collected);
    let offset = 0;
    for (const chunk of this.chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.chunks = [];
    this.collected = 0;
    this.onWindow(encodeWav(merged, this.ctx.sampleRate), this.ctx.sampleRate);
  }

  async close() {
    this.processor.onaudioprocess = null;
    this.source.disconnect();
    this.processor.disconnect();
    await this.ctx.close().catch(() => undefined);
  }
}

/** One JPEG frame from a playing <video>, or null if it has no pixels yet. */
export async function grabFrame(video: HTMLVideoElement, maxEdge = 640): Promise<Blob | null> {
  if (!video.videoWidth || !video.videoHeight) return null;

  const scale = Math.min(1, maxEdge / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.85));
}
