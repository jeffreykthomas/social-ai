/**
 * ElevenLabs Conversational AI WebSocket client.
 *
 * Handles:
 *  - WebSocket connection to ElevenLabs (via signed URL from our API)
 *  - Mic capture → PCM 16-bit 16 kHz → base64 → WebSocket
 *  - WebSocket audio events → PCM decode → queued AudioContext playback
 *  - Transcript and agent-response forwarding
 */

import { appEnv } from '../../../core/config/env';

// ── Types ───────────────────────────────────────────────────────────────────

export type ConvaiEvents = {
  onConnected?: () => void;
  onDisconnected?: () => void;
  onUserTranscript?: (text: string) => void;
  onAgentResponse?: (text: string) => void;
  onAgentResponseCorrection?: (original: string, corrected: string) => void;
  /** Called progressively as audio chunks arrive — text synced to audio like captions. */
  onTextReveal?: (text: string) => void;
  onSpeakingStart?: () => void;
  onSpeakingEnd?: () => void;
  onError?: (error: string) => void;
};

type ElevenLabsMessage =
  | { type: 'conversation_initiation_metadata'; conversation_initiation_metadata_event: unknown }
  | { type: 'user_transcript'; user_transcription_event: { user_transcript: string } }
  | { type: 'agent_response'; agent_response_event: { agent_response: string } }
  | {
      type: 'agent_response_correction';
      agent_response_correction_event: {
        original_agent_response: string;
        corrected_agent_response: string;
      };
    }
  | {
      type: 'audio';
      audio_event: {
        audio_base_64: string;
        event_id: number;
        alignment?: {
          chars: string[];
          char_durations_ms: number[];
          char_start_times_ms: number[];
        };
      };
    }
  | { type: 'interruption'; interruption_event: { reason: string } }
  | { type: 'ping'; ping_event: { event_id: number; ping_ms?: number } }
  | { type: 'vad_score'; vad_event: unknown }
  | { type: 'internal_tentative_agent_response'; tentative_agent_response_internal_event: unknown };

// ── Audio helpers ───────────────────────────────────────────────────────────

const TARGET_SAMPLE_RATE = 16_000;

function downsample(input: Float32Array, srcRate: number, dstRate: number): Int16Array {
  const ratio = srcRate / dstRate;
  const length = Math.floor(input.length / ratio);
  const output = new Int16Array(length);

  for (let i = 0; i < length; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const frac = srcIdx - lo;
    const sample = lo + 1 < input.length ? input[lo] * (1 - frac) + input[lo + 1] * frac : input[lo];
    output[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
  }

  return output;
}

function int16ToBase64(arr: Int16Array): string {
  const bytes = new Uint8Array(arr.buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToInt16(b64: string): Int16Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

// ── Inline AudioWorklet processor (avoids a separate file) ──────────────────

const MIC_WORKLET_CODE = `
class MicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input?.length) {
      this.port.postMessage(input);
    }
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);
`;

function createWorkletBlobUrl(): string {
  const blob = new Blob([MIC_WORKLET_CODE], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}

// ── ConvaiClient ────────────────────────────────────────────────────────────

export class ConvaiClient {
  private ws: WebSocket | null = null;
  private micStream: MediaStream | null = null;
  private captureCtx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private workletBlobUrl: string | null = null;
  private playbackCtx: AudioContext | null = null;
  private nextPlayTime = 0;
  private activeSources: AudioBufferSourceNode[] = [];
  private events: ConvaiEvents;
  private _connected = false;
  private _speaking = false;
  private _pendingAgentText = '';
  private _revealedChars = 0;

  constructor(events: ConvaiEvents) {
    this.events = events;
  }

  get connected(): boolean {
    return this._connected;
  }
  get speaking(): boolean {
    return this._speaking;
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /** The voice ID used for this session (stable after first start). */
  get resolvedVoiceId(): string | undefined {
    return this._resolvedVoiceId;
  }
  private _resolvedVoiceId?: string;

  async start(
    sessionId: string,
    options?: { mic?: boolean; gender?: string; voiceId?: string; voiceName?: string }
  ): Promise<void> {
    if (this._connected) return;
    const enableMic = options?.mic ?? true;

    // 1. Get signed URL from backend (pass sessionId + gender for voice matching)
    const qp = new URLSearchParams();
    qp.set('sessionId', sessionId);
    if (options?.gender) qp.set('gender', options.gender);
    const res = await fetch(`${appEnv.apiBaseUrl}/voice/convai/signed-url?${qp.toString()}`);
    if (!res.ok) throw new Error(`Failed to get signed URL (${res.status})`);
    const data = (await res.json()) as { signedUrl: string; voiceId?: string };
    const signedUrl = data.signedUrl;

    // Use cached voice ID if provided, otherwise use the one from backend
    const voiceId = options?.voiceId ?? data.voiceId;
    this._resolvedVoiceId = voiceId;

    // 2. Open WebSocket
    this.ws = new WebSocket(signedUrl);

    this.ws.onopen = () => {
      this._connected = true;
      this.events.onConnected?.();

      // Send init — pass sessionId as user_id so it reaches our LLM proxy.
      // Flat structure: properties are siblings of type, not nested.
      this.ws!.send(
        JSON.stringify({
          type: 'conversation_initiation_client_data',
          user_id: sessionId
        })
      );

      // Only start mic capture if requested (voice_voice needs it, text_voice doesn't)
      if (enableMic) {
        void this.startMicCapture();
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as ElevenLabsMessage;
        this.handleMessage(msg);
      } catch {
        // ignore parse errors
      }
    };

    this.ws.onclose = () => {
      this._connected = false;
      this.cleanup();
      this.events.onDisconnected?.();
    };

    this.ws.onerror = () => {
      this.events.onError?.('WebSocket connection error');
    };

    // 3. Init playback context
    this.playbackCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    this.nextPlayTime = 0;
  }

  stop(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.cleanup();
  }

  /** Send a text message through the ConvAI pipeline (skips STT, goes straight to LLM + TTS). */
  sendText(text: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: 'user_message', text }));
  }

  // ── Message handling ────────────────────────────────────────────────────

  private handleMessage(msg: ElevenLabsMessage): void {
    switch (msg.type) {
      case 'ping':
        if (this.ws?.readyState === WebSocket.OPEN) {
          const delay = msg.ping_event.ping_ms ?? 0;
          setTimeout(() => {
            this.ws?.send(JSON.stringify({ type: 'pong', event_id: msg.ping_event.event_id }));
          }, delay);
        }
        break;

      case 'user_transcript':
        this.events.onUserTranscript?.(msg.user_transcription_event.user_transcript);
        break;

      case 'agent_response':
        // Store full text but don't reveal yet — text will sync with audio chunks
        this._pendingAgentText = msg.agent_response_event.agent_response;
        this._revealedChars = 0;
        this.events.onAgentResponse?.(msg.agent_response_event.agent_response);
        break;

      case 'agent_response_correction':
        this._pendingAgentText = msg.agent_response_correction_event.corrected_agent_response;
        this._revealedChars = 0;
        this.events.onAgentResponseCorrection?.(
          msg.agent_response_correction_event.original_agent_response,
          msg.agent_response_correction_event.corrected_agent_response
        );
        break;

      case 'audio':
        this.revealTextForChunk(msg.audio_event.alignment);
        this.enqueueAudio(msg.audio_event.audio_base_64);
        break;

      case 'interruption':
        this.flushPendingText();
        this.clearPlaybackQueue();
        break;

      default:
        // conversation_initiation_metadata, vad_score, tentative — ignore
        break;
    }
  }

  // ── Mic capture ─────────────────────────────────────────────────────────

  private async startMicCapture(): Promise<void> {
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      this.events.onError?.('Could not access microphone');
      return;
    }

    this.captureCtx = new AudioContext();
    const srcRate = this.captureCtx.sampleRate;

    // Register inline AudioWorklet processor
    this.workletBlobUrl = createWorkletBlobUrl();
    await this.captureCtx.audioWorklet.addModule(this.workletBlobUrl);

    const source = this.captureCtx.createMediaStreamSource(this.micStream);
    this.workletNode = new AudioWorkletNode(this.captureCtx, 'mic-processor');

    this.workletNode.port.onmessage = (e: MessageEvent<Float32Array>) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

      const pcm16 = downsample(e.data, srcRate, TARGET_SAMPLE_RATE);
      const base64 = int16ToBase64(pcm16);
      this.ws.send(JSON.stringify({ user_audio_chunk: base64 }));
    };

    source.connect(this.workletNode);
    this.workletNode.connect(this.captureCtx.destination);
  }

  // ── Text reveal (synced to audio) ───────────────────────────────────────

  private revealTextForChunk(alignment?: { chars: string[] }): void {
    if (!this._pendingAgentText) return;

    if (alignment?.chars?.length) {
      this._revealedChars += alignment.chars.length;
    } else {
      // No alignment data — reveal all remaining text as fallback
      this._revealedChars = this._pendingAgentText.length;
    }

    const revealed = this._pendingAgentText.slice(0, this._revealedChars);
    this.events.onTextReveal?.(revealed);
  }

  private flushPendingText(): void {
    if (this._pendingAgentText && this._revealedChars < this._pendingAgentText.length) {
      this.events.onTextReveal?.(this._pendingAgentText);
    }
    this._pendingAgentText = '';
    this._revealedChars = 0;
  }

  // ── Audio playback ──────────────────────────────────────────────────────

  private enqueueAudio(base64: string): void {
    if (!this.playbackCtx) return;

    const pcm16 = base64ToInt16(base64);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 32768;
    }

    const buffer = this.playbackCtx.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
    buffer.getChannelData(0).set(float32);

    const sourceNode = this.playbackCtx.createBufferSource();
    sourceNode.buffer = buffer;
    sourceNode.connect(this.playbackCtx.destination);

    const now = this.playbackCtx.currentTime;
    const startTime = Math.max(now + 0.02, this.nextPlayTime); // 20ms buffer
    sourceNode.start(startTime);
    this.nextPlayTime = startTime + buffer.duration;

    // Track speaking state
    if (!this._speaking) {
      this._speaking = true;
      this.events.onSpeakingStart?.();
    }

    sourceNode.onended = () => {
      const idx = this.activeSources.indexOf(sourceNode);
      if (idx >= 0) this.activeSources.splice(idx, 1);

      if (this.activeSources.length === 0) {
        this.flushPendingText();
        this._speaking = false;
        this.events.onSpeakingEnd?.();
      }
    };

    this.activeSources.push(sourceNode);
  }

  private clearPlaybackQueue(): void {
    for (const src of this.activeSources) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
    }
    this.activeSources = [];
    this.nextPlayTime = 0;

    if (this._speaking) {
      this._speaking = false;
      this.events.onSpeakingEnd?.();
    }
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  private cleanup(): void {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.workletBlobUrl) {
      URL.revokeObjectURL(this.workletBlobUrl);
      this.workletBlobUrl = null;
    }
    if (this.captureCtx) {
      void this.captureCtx.close();
      this.captureCtx = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }

    this.clearPlaybackQueue();

    if (this.playbackCtx) {
      void this.playbackCtx.close();
      this.playbackCtx = null;
    }

    this._connected = false;
  }
}
