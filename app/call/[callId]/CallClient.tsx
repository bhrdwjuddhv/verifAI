'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { PhoneOff, Video, VideoOff, Mic, MicOff, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { ConsentGate } from '@/components/call/ConsentGate';
import { VoiceRiskMeter } from '@/components/call/VoiceRiskMeter';
import { CallTimeline, TimelineWindow } from '@/components/call/CallTimeline';
import { VoiceAlert } from '@/components/call/VoiceAlert';
import { AudioWindower, grabFrame } from '@/lib/call/capture';
import {
  initialState,
  parseCallWeights,
  pushWindow,
  type RiskState,
} from '@/lib/call/risk';

const ICE: RTCConfiguration = {
  // Public STUN only. No TURN, so a call between two symmetric NATs will fail to connect —
  // that is a limitation of running this with zero infrastructure, not a bug to hide.
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

const WINDOW_SECONDS = 3;
const FRAME_EVERY_MS = 6000;
const POLL_MS = 700;

// Windows that must agree before the displayed band moves, and the EMA weight of the newest
// window. Both are the difference between a guard people trust and one they mute.
const HYSTERESIS = 3;
const ALPHA = 0.4;
const CALL_WEIGHTS = parseCallWeights(process.env.NEXT_PUBLIC_CALL_FUSION_WEIGHTS);

type Phase = 'idle' | 'connecting' | 'connected' | 'ended' | 'error';

export function CallClient({ callId, role }: { callId: string; role: 'caller' | 'callee' }) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [monitoring, setMonitoring] = useState<boolean | null>(null); // null = not asked yet
  const [videoOn, setVideoOn] = useState(true);
  const [micOn, setMicOn] = useState(true);
  const [copied, setCopied] = useState(false);

  const [windows, setWindows] = useState<TimelineWindow[]>([]);
  const [risk, setRisk] = useState<RiskState>(initialState());
  const [modelSource, setModelSource] = useState<string | null>(null);
  const [alertDismissed, setAlertDismissed] = useState(false);
  const [faceScore, setFaceScore] = useState<number | null>(null);
  const [monitorNote, setMonitorNote] = useState<string | null>(null);
  // Latest video score, folded into the next audio window rather than shown on its own: one
  // fused number per window is the whole point of the call profile.
  const latestVideo = useRef<number | null>(null);

  const localVideo = useRef<HTMLVideoElement>(null);
  const remoteVideo = useRef<HTMLVideoElement>(null);
  const pc = useRef<RTCPeerConnection | null>(null);
  const peerId = useRef<string>('');
  const since = useRef(0);
  const windower = useRef<AudioWindower | null>(null);
  const started = useRef(0);
  const stopped = useRef(false);

  // ---------------------------------------------------------------- signalling
  const send = useCallback(
    async (data: unknown) => {
      await fetch(`/api/call/${callId}/signal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ peerId: peerId.current, data }),
      }).catch(() => undefined);
    },
    [callId]
  );

  // ---------------------------------------------------------------- monitoring
  const scoreWindow = useCallback(async (wav: Blob) => {
    const form = new FormData();
    form.append('file', wav, 'window.wav');
    const res = await fetch('/api/live/audio-window', { method: 'POST', body: form });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      // Honest: say the detector is unavailable rather than showing a stale or invented score.
      setMonitorNote(body.detail || body.error || `Detector unavailable (HTTP ${res.status}).`);
      // The band is left where it was: the detector going away is not evidence about the call.
      return;
    }

    const out = await res.json();
    setMonitorNote(null);
    if (out.modelSource) setModelSource(out.modelSource);

    const t = (Date.now() - started.current) / 1000;
    if (!out.speechDetected || out.fakeProbability === null) {
      // Silence holds the band where it is. It is not evidence that the call is fine.
      setRisk((state) => pushWindow(state, { audio: null, video: null }, {
        alpha: ALPHA,
        hysteresis: HYSTERESIS,
        weights: CALL_WEIGHTS,
      }));
      setWindows((w) => [...w, { t, fake: null, speech: false }]);
      return;
    }

    const fake: number = out.fakeProbability;
    setWindows((w) => [...w, { t, fake, speech: true }]);
    setRisk((state) =>
      pushWindow(state, { audio: fake, video: latestVideo.current }, {
        alpha: ALPHA,
        hysteresis: HYSTERESIS,
        weights: CALL_WEIGHTS,
      })
    );
  }, []);

  const scoreFrame = useCallback(async () => {
    const video = remoteVideo.current;
    if (!video) return;
    const frame = await grabFrame(video);
    if (!frame) return;

    const form = new FormData();
    form.append('file', frame, 'frame.jpg');
    const res = await fetch('/api/live/frame', { method: 'POST', body: form });
    if (!res.ok) return;
    const out = await res.json();
    setFaceScore(out.fakeProbability ?? null);
    latestVideo.current = out.fakeProbability ?? null;
    if (out.modelSource) setModelSource((s) => (s ? `${s} + ${out.modelSource}` : out.modelSource));
  }, []);

  // ---------------------------------------------------------------- connect
  useEffect(() => {
    let poll: ReturnType<typeof setInterval> | null = null;
    let frames: ReturnType<typeof setInterval> | null = null;

    (async () => {
      try {
        setPhase('connecting');
        const stored = sessionStorage.getItem(`verifai-call-${callId}`);
        if (!stored) throw new Error('This call was not joined from the lobby. Open /call and join again.');
        peerId.current = stored;

        const local = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        if (localVideo.current) localVideo.current.srcObject = local;

        const conn = new RTCPeerConnection(ICE);
        pc.current = conn;
        local.getTracks().forEach((track) => conn.addTrack(track, local));

        conn.onicecandidate = (e) => {
          if (e.candidate) void send({ kind: 'ice', candidate: e.candidate.toJSON() });
        };
        conn.ontrack = (e) => {
          const [remote] = e.streams;
          if (remoteVideo.current) remoteVideo.current.srcObject = remote;
          setPhase('connected');
          started.current = Date.now();

          if (monitoring && !windower.current) {
            windower.current = new AudioWindower(remote, WINDOW_SECONDS, (wav) => void scoreWindow(wav));
            frames = setInterval(() => void scoreFrame(), FRAME_EVERY_MS);
          }
        };

        if (role === 'caller') {
          const offer = await conn.createOffer();
          await conn.setLocalDescription(offer);
          await send({ kind: 'offer', sdp: offer });
        }

        poll = setInterval(async () => {
          if (stopped.current) return;
          const res = await fetch(
            `/api/call/${callId}/signal?peerId=${peerId.current}&since=${since.current}`
          ).catch(() => null);
          if (!res?.ok) return;
          const { messages, last } = await res.json();
          since.current = last ?? since.current;

          for (const m of messages ?? []) {
            const data = m.data;
            if (data.kind === 'offer') {
              await conn.setRemoteDescription(new RTCSessionDescription(data.sdp));
              const answer = await conn.createAnswer();
              await conn.setLocalDescription(answer);
              await send({ kind: 'answer', sdp: answer });
            } else if (data.kind === 'answer') {
              if (!conn.currentRemoteDescription) {
                await conn.setRemoteDescription(new RTCSessionDescription(data.sdp));
              }
            } else if (data.kind === 'ice') {
              await conn.addIceCandidate(new RTCIceCandidate(data.candidate)).catch(() => undefined);
            } else if (data.kind === 'bye') {
              setPhase('ended');
            }
          }
        }, POLL_MS);
      } catch (err: any) {
        setError(err?.message || 'Could not start the call.');
        setPhase('error');
      }
    })();

    return () => {
      stopped.current = true;
      if (poll) clearInterval(poll);
      if (frames) clearInterval(frames);
      void windower.current?.close();
      pc.current?.close();
    };
    // monitoring is read when the remote track arrives; changing it mid-call is not supported.
  }, [callId, role, monitoring, send, scoreWindow, scoreFrame]);

  const hangUp = async () => {
    stopped.current = true;
    await send({ kind: 'bye' });
    await fetch('/api/call/end', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ callId }),
    }).catch(() => undefined);
    void windower.current?.close();
    pc.current?.close();
    setPhase('ended');
  };

  const toggleTrack = (kind: 'audio' | 'video') => {
    const stream = localVideo.current?.srcObject as MediaStream | null;
    stream?.getTracks().filter((t) => t.kind === kind).forEach((t) => (t.enabled = !t.enabled));
    if (kind === 'audio') setMicOn((v) => !v);
    else setVideoOn((v) => !v);
  };

  if (monitoring === null) {
    return (
      <div className="max-w-xl mx-auto py-10 px-4 space-y-4">
        <h1 className="text-2xl font-bold text-white">Joining call {callId}</h1>
        <ConsentGate onAccept={() => setMonitoring(true)} onDecline={() => setMonitoring(false)} />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-white">Call {callId}</h1>
          <p className="text-xs text-ink-400 font-normal">
            {phase === 'connected' ? 'Connected' : phase === 'connecting' ? 'Waiting for the other side…' : phase}
            {monitoring ? ' · monitoring on' : ' · monitoring off'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            leftIcon={copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            onClick={() => {
              navigator.clipboard?.writeText(`${location.origin}/call?join=${callId}`);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
          >
            {copied ? 'Copied' : 'Copy invite link'}
          </Button>
          <Button variant="ghost" size="sm" leftIcon={<PhoneOff className="w-4 h-4" />} onClick={hangUp}>
            Hang up
          </Button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="relative rounded-2xl overflow-hidden border border-ink-800 bg-black aspect-video">
            <video ref={remoteVideo} autoPlay playsInline className="w-full h-full object-cover" />
            <span className="absolute top-3 left-3 text-[11px] px-2 py-1 rounded-full bg-black/60 text-white">
              Remote {monitoring && '· analysed'}
            </span>
            <video
              ref={localVideo}
              autoPlay
              playsInline
              muted
              className="absolute bottom-3 right-3 w-40 rounded-xl border border-ink-700"
            />
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              leftIcon={micOn ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4 text-rose-400" />}
              onClick={() => toggleTrack('audio')}
            >
              {micOn ? 'Mute' : 'Unmute'}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={videoOn ? <Video className="w-4 h-4" /> : <VideoOff className="w-4 h-4 text-rose-400" />}
              onClick={() => toggleTrack('video')}
            >
              {videoOn ? 'Stop video' : 'Start video'}
            </Button>
          </div>
        </div>

        <div className="space-y-4">
          {monitoring ? (
            <>
              <VoiceAlert
                visible={risk.consecutiveHigh >= HYSTERESIS && !alertDismissed}
                consecutive={risk.consecutiveHigh}
                onDismiss={() => setAlertDismissed(true)}
              />
              <VoiceRiskMeter
                trust={risk.trust}
                band={risk.band}
                windowsScored={risk.scored}
                windowsSkipped={risk.skipped}
                modelSource={modelSource}
                pending={
                  risk.pendingCount > 0 && risk.pendingBand !== risk.band
                    ? { band: risk.pendingBand, count: risk.pendingCount, needed: HYSTERESIS }
                    : null
                }
                fusedWith={risk.scored > 0 && latestVideo.current !== null ? 'audio + video' : 'audio'}
              />
              {faceScore !== null && (
                <div className="rounded-2xl border border-ink-800 bg-ink-900/60 p-4 text-xs text-ink-300">
                  <span className="font-semibold text-white">Video frame:</span> {faceScore}% synthetic
                  <p className="text-[11px] text-ink-500 mt-1 font-normal">
                    Sampled every {FRAME_EVERY_MS / 1000}s from the remote tile.
                  </p>
                </div>
              )}
              {monitorNote && (
                <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
                  {monitorNote}
                </div>
              )}
              <CallTimeline windows={windows} />
            </>
          ) : (
            <div className="rounded-2xl border border-ink-800 bg-ink-900/60 p-5 text-xs text-ink-400">
              Monitoring is off for this call. Nothing is analysed and nothing leaves your device
              beyond the call itself.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
