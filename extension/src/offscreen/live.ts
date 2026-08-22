/**
 * Tab audio capture and windowing, in the offscreen document.
 *
 * The service worker cannot do this: `getUserMedia` and `AudioContext` need a document, and a
 * worker is killed after ~30s idle — a call lasts minutes. The worker gets the stream ID and
 * hands it here; the audio itself never leaves this document except as 3-second WAV windows.
 */

const SAMPLE_RATE = 16000;

export interface LiveCaptureHandlers {
  /**
   * One completed window, in both forms the two paths need: raw mono samples for the
   * on-device chain, and WAV for the backend. The samples are handed over rather than
   * re-decoded from the WAV so the on-device path is not quietly scoring a 16-bit
   * round-trip of what the server would have seen.
   */
  onWindow: (wav: ArrayBuffer, samples: Float32Array, sampleRate: number) => void;
  onError: (message: string) => void;
}

let context: AudioContext | null = null;
let stream: MediaStream | null = null;
let processor: ScriptProcessorNode | null = null;

/** Float32 mono -> 16-bit PCM WAV. The service decodes WAV and raw PCM only. */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

export async function startCapture(
  streamId: string,
  windowSeconds: number,
  handlers: LiveCaptureHandlers
): Promise<void> {
  await stopCapture();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // The tabCapture handshake: the worker mints the id, this document redeems it.
        mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId },
      },
      video: false,
    } as MediaStreamConstraints);
  } catch (err) {
    handlers.onError(`Could not capture tab audio: ${(err as Error)?.message ?? err}`);
    return;
  }

  context = new AudioContext({ sampleRate: SAMPLE_RATE });
  const source = context.createMediaStreamSource(stream);
  processor = context.createScriptProcessor(4096, 1, 1);

  const target = Math.round(context.sampleRate * windowSeconds);
  let chunks: Float32Array[] = [];
  let collected = 0;

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
    collected += input.length;
    if (collected < target) return;

    const merged = new Float32Array(collected);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    chunks = [];
    collected = 0;
    handlers.onWindow(encodeWav(merged, context!.sampleRate), merged, context!.sampleRate);
  };

  source.connect(processor);

  // Capturing a tab MUTES it unless the stream is played back. Routing it to the destination
  // at unity gain gives the user their call audio back — without this the extension silences
  // the very call it is guarding.
  const passthrough = context.createGain();
  passthrough.gain.value = 1;
  source.connect(passthrough);
  passthrough.connect(context.destination);

  // The ScriptProcessor needs a path to the destination to be pulled at all; a muted branch
  // keeps it running without duplicating the audio.
  const silent = context.createGain();
  silent.gain.value = 0;
  processor.connect(silent);
  silent.connect(context.destination);
}

export async function stopCapture(): Promise<void> {
  if (processor) {
    processor.onaudioprocess = null;
    processor.disconnect();
    processor = null;
  }
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  if (context) {
    await context.close().catch(() => undefined);
    context = null;
  }
}

export function captureActive(): boolean {
  return stream !== null;
}
