<script setup lang="ts">
import type { NeedDeltaState, RunReportDimensionSummary } from '@social-pet/domain';
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';

import { useSessionStore } from '../features/pet-core/stores/sessionStore';
import AudioAura3D from '../scenes/audio-aura/AudioAura3D.vue';
import { appEnv } from '../core/config/env';
import { ConvaiClient } from '../features/pet-core/services/convaiClient';

const store = useSessionStore();

const speakerOn = ref(false);  // AI voice output (ConvAI TTS)
const micOn = ref(false);       // User mic input
const isRecording = ref(false);
const convaiActive = ref(false);
let convaiClient: ConvaiClient | null = null;
let sessionVoiceId: string | null = null; // Cached for the session so voice doesn't change
const micLevel = ref(0);
const speechLevel = ref(0);
const isSpeaking = ref(false);
const transcriptFinal = ref('');
const transcriptInterim = ref('');
const voiceError = ref<string | null>(null);
const voiceSupported = ref(false);

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let currentAudio: HTMLAudioElement | null = null;
let mediaStream: MediaStream | null = null;
let audioContext: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let micFrame = 0;
let speakingInterval: number | null = null;
let speakingPhase = 0;
let lastSpokenEventId: string | null = null;

const chatScroll = ref<HTMLElement | null>(null);
const pendingBubble = ref<HTMLElement | null>(null);
const debugOpen = ref(false);

// Collapsing aura header — shrinks from 300→100 as chat scrolls
const AURA_MAX_HEIGHT = 300;
const AURA_MIN_HEIGHT = 100;
const AURA_SCROLL_RANGE = 200;
const auraHeight = ref(AURA_MAX_HEIGHT);

const needsVoiceInput = computed(() => micOn.value);
const needsVoiceOutput = computed(() => speakerOn.value);

const auraLevel = computed(() => {
  const base = store.isThinking ? 0.14 : 0.04;
  return Math.max(base, micLevel.value, speechLevel.value, store.isTalking ? 0.25 : 0);
});

const reportDimensions = computed(() => {
  if (!store.runReport) return [];
  return [...store.runReport.dimensions].sort((a, b) => b.score - a.score);
});

function formatDimensionKey(key: RunReportDimensionSummary['key']): string {
  return key.replaceAll('_', ' ');
}

function toggleSpeaker(): void {
  speakerOn.value = !speakerOn.value;

  if (speakerOn.value) {
    // Connect ConvAI with mic if micOn, without if not
    void startConvai({ mic: micOn.value });
  } else {
    stopConvai();
  }
}

function toggleMic(): void {
  micOn.value = !micOn.value;

  if (micOn.value) {
    if (convaiActive.value) {
      // ConvAI is already running (speaker on) — restart with mic enabled
      stopConvai();
      void startConvai({ mic: true });
    } else {
      // No ConvAI — use manual record + Whisper STT
      void startVoiceInput();
    }
  } else {
    if (convaiActive.value) {
      // ConvAI running — restart without mic
      stopConvai();
      void startConvai({ mic: false });
    } else {
      void stopVoiceInput(false);
    }
  }
}

function deltaSummary(deltas: NeedDeltaState): string {
  return Object.entries(deltas)
    .map(([k, v]) => `${k} ${v! >= 0 ? '+' : ''}${v!.toFixed(2)}`)
    .join(' · ');
}

function initVoice(): void {
  voiceSupported.value = !!(navigator.mediaDevices?.getUserMedia);
}

// Map persona gender to one of the named voices on the ElevenLabs agent.
// Names must match the lowercase voice names configured on the agent.
const FEMALE_VOICES = ['mariana', 'jane', 'primary'];
const MALE_VOICES = ['peter', 'julian', 'jon'];

function pickVoiceName(gender?: string): string | undefined {
  const g = (gender ?? '').toLowerCase();
  const isFemale = g.includes('woman') || g.includes('female') || g.includes('she');
  const pool = isFemale ? FEMALE_VOICES : MALE_VOICES;
  return pool[Math.floor(Math.random() * pool.length)];
}

// ── ElevenLabs Conversational AI (real-time voice) ────────────────────────

async function startConvai(options?: { mic?: boolean }): Promise<void> {
  if (convaiActive.value || !store.sessionId) return;

  voiceError.value = null;

  convaiClient = new ConvaiClient({
    onConnected: () => {
      convaiActive.value = true;
      if (options?.mic) {
        isRecording.value = true;
        transcriptInterim.value = 'Listening...';
      }
    },
    onDisconnected: () => {
      convaiActive.value = false;
      isRecording.value = false;
      transcriptInterim.value = '';
    },
    onUserTranscript: (text) => {
      transcriptFinal.value = text;
      transcriptInterim.value = '';
    },
    onAgentResponse: () => {
      // Don't show text here — it reveals in sync with audio via onTextReveal.
      // Clear pending user message and refresh game state from backend.
      store.pendingUserMessage = null;
      void store.refreshSessionState();
    },
    onAgentResponseCorrection: () => {
      // Corrected text will flow through onTextReveal with audio alignment
    },
    onTextReveal: (text) => {
      store.streamingReply = text;
    },
    onSpeakingStart: () => {
      isSpeaking.value = true;
      startSpeakingAnimation();
    },
    onSpeakingEnd: () => {
      isSpeaking.value = false;
      stopSpeakingAnimation();
      store.streamingReply = '';
      scrollToLatestUserMessage();
    },
    onError: (err) => {
      voiceError.value = err;
    }
  });

  try {
    await convaiClient.start(store.sessionId, {
      mic: options?.mic ?? true,
      gender: store.activePersona?.gender,
      voiceId: sessionVoiceId ?? undefined,
      voiceName: sessionVoiceId ? undefined : pickVoiceName(store.activePersona?.gender)
    });
    // Cache the resolved voice ID so reconnects use the same voice
    if (convaiClient.resolvedVoiceId) {
      sessionVoiceId = convaiClient.resolvedVoiceId;
    }
  } catch (err) {
    voiceError.value = err instanceof Error ? err.message : 'Could not start conversation.';
    convaiActive.value = false;
  }
}

function stopConvai(): void {
  convaiClient?.stop();
  convaiClient = null;
  convaiActive.value = false;
  isRecording.value = false;
  isSpeaking.value = false;
  stopSpeakingAnimation();
  transcriptFinal.value = '';
  transcriptInterim.value = '';
}

function startSpeakingAnimation(): void {
  if (speakingInterval !== null) {
    window.clearInterval(speakingInterval);
  }

  speakingPhase = 0;
  speakingInterval = window.setInterval(() => {
    speakingPhase += 0.22;
    speechLevel.value = 0.35 + Math.abs(Math.sin(speakingPhase)) * 0.55;
  }, 55);
}

function stopSpeakingAnimation(): void {
  if (speakingInterval !== null) {
    window.clearInterval(speakingInterval);
    speakingInterval = null;
  }
  speechLevel.value = 0;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result as string;
      resolve(dataUrl.split(',')[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function transcribeAudio(blob: Blob): Promise<string> {
  const base64 = await blobToBase64(blob);

  const res = await fetch(`${appEnv.apiBaseUrl}/voice/transcribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio: base64,
      mimeType: blob.type || 'audio/webm'
    })
  });

  if (!res.ok) throw new Error(`Transcription failed (${res.status})`);

  const data = (await res.json()) as { text: string };
  return data.text;
}

async function playVoiceResponse(text: string): Promise<void> {
  if (!needsVoiceOutput.value || !text.trim()) return;

  // Stop any currently playing audio
  if (currentAudio) {
    currentAudio.pause();
    if (currentAudio.src) URL.revokeObjectURL(currentAudio.src);
    currentAudio = null;
  }

  isSpeaking.value = true;
  startSpeakingAnimation();

  try {
    const res = await fetch(`${appEnv.apiBaseUrl}/voice/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) throw new Error(`Speech synthesis failed (${res.status})`);

    const audioBlob = await res.blob();
    const url = URL.createObjectURL(audioBlob);
    const audio = new Audio(url);
    currentAudio = audio;

    audio.onended = () => {
      isSpeaking.value = false;
      stopSpeakingAnimation();
      URL.revokeObjectURL(url);
      if (currentAudio === audio) currentAudio = null;
    };

    audio.onerror = () => {
      isSpeaking.value = false;
      stopSpeakingAnimation();
      URL.revokeObjectURL(url);
      if (currentAudio === audio) currentAudio = null;
    };

    await audio.play();
  } catch {
    isSpeaking.value = false;
    stopSpeakingAnimation();
  }
}

function stopMicMeter(): void {
  cancelAnimationFrame(micFrame);

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    void audioContext.close();
    audioContext = null;
  }

  analyser = null;
  micLevel.value = 0;
}

async function startMicMeter(): Promise<void> {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;

  const source = audioContext.createMediaStreamSource(mediaStream);
  source.connect(analyser);

  const data = new Uint8Array(analyser.fftSize);

  const tick = () => {
    if (!analyser) return;

    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i += 1) {
      const normalized = (data[i] - 128) / 128;
      sum += normalized * normalized;
    }

    const rms = Math.sqrt(sum / data.length);
    micLevel.value = Math.min(1, rms * 3.2);

    micFrame = requestAnimationFrame(tick);
  };

  tick();
}

async function startVoiceInput(): Promise<void> {
  voiceError.value = null;

  if (!voiceSupported.value) {
    voiceError.value = 'Voice input is not supported in this browser.';
    return;
  }

  if (isRecording.value) return;

  transcriptFinal.value = '';
  transcriptInterim.value = '';

  try {
    await startMicMeter();
  } catch (error) {
    voiceError.value = error instanceof Error ? error.message : 'Could not access microphone.';
    return;
  }

  // Start MediaRecorder using the mic stream from startMicMeter
  audioChunks = [];
  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : 'audio/webm';
  mediaRecorder = new MediaRecorder(mediaStream!, { mimeType });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };

  mediaRecorder.start(250);
  isRecording.value = true;
  transcriptInterim.value = 'Listening...';
}

async function stopVoiceInput(sendCapturedText = true): Promise<void> {
  if (!isRecording.value) return;
  isRecording.value = false;

  await new Promise<void>((resolve) => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      stopMicMeter();
      transcriptInterim.value = '';
      resolve();
      return;
    }

    mediaRecorder.onstop = async () => {
      stopMicMeter();

      if (!sendCapturedText || audioChunks.length === 0) {
        transcriptInterim.value = '';
        resolve();
        return;
      }

      transcriptInterim.value = 'Transcribing...';

      try {
        const blob = new Blob(audioChunks, { type: mediaRecorder!.mimeType });
        const text = await transcribeAudio(blob);
        transcriptFinal.value = text;
        transcriptInterim.value = '';

        if (text.trim()) {
          await store.sendMessage(text.trim());
        }
      } catch (err) {
        voiceError.value = err instanceof Error ? err.message : 'Transcription failed.';
        transcriptInterim.value = '';
      }

      resolve();
    };

    mediaRecorder.stop();
  });

  mediaRecorder = null;
}

async function sendCurrentText(): Promise<void> {
  // When ConvAI is active, route through ConvAI only.
  // The ConvAI LLM proxy already runs the full game pipeline (Claude + state updates).
  // Don't double-send through the normal flow — it would race and show text before audio.
  if (convaiActive.value && convaiClient) {
    const text = store.messageDraft.trim();
    if (!text) return;
    store.messageDraft = '';
    store.pendingUserMessage = text;
    convaiClient.sendText(text);
    return;
  }

  await store.sendMessage();
}

function stopCurrentOutput(): void {
  store.stopCurrentStream();

  // Stop ConvAI playback
  if (convaiActive.value) {
    stopConvai();
  }

  if (currentAudio) {
    currentAudio.pause();
    if (currentAudio.src) URL.revokeObjectURL(currentAudio.src);
    currentAudio = null;
  }
  isSpeaking.value = false;
  stopSpeakingAnimation();
}

async function allowOutreachContact(): Promise<void> {
  await store.setOutreachPreference('granted', store.outreachContactDraft || undefined);
  await store.markOutreachSent();
}

async function declineOutreachContact(): Promise<void> {
  await store.setOutreachPreference('declined');
}

async function acknowledgeOutreach(): Promise<void> {
  await store.markOutreachSent();
}

function scrollToLatestUserMessage(): void {
  void nextTick(() => {
    if (pendingBubble.value) {
      pendingBubble.value.scrollIntoView({ block: 'start', behavior: 'smooth' });
      return;
    }
    // Fall back: scroll the last user bubble into view at top
    if (chatScroll.value) {
      const userBubbles = chatScroll.value.querySelectorAll('.chat-user');
      const last = userBubbles[userBubbles.length - 1];
      if (last) {
        last.scrollIntoView({ block: 'start', behavior: 'smooth' });
        return;
      }
      chatScroll.value.scrollTop = chatScroll.value.scrollHeight;
    }
  });
}

function onChatScroll(): void {
  const scrollTop = chatScroll.value?.scrollTop ?? 0;
  const t = Math.min(1, scrollTop / AURA_SCROLL_RANGE);
  auraHeight.value = Math.round(AURA_MAX_HEIGHT - t * (AURA_MAX_HEIGHT - AURA_MIN_HEIGHT));
}

// TTS: fire as soon as full text is available (before reveal animation finishes)
// Skip when ConvAI is active — it handles its own audio pipeline.
watch(
  () => store.pendingTtsText,
  (entry) => {
    if (!entry) return;
    if (convaiActive.value) return;
    if (entry.id === lastSpokenEventId) return;

    lastSpokenEventId = entry.id;
    if (needsVoiceOutput.value) {
      void playVoiceResponse(entry.text);
    }
  }
);

// Scroll when an event commits (after reveal finishes)
watch(
  () => {
    const e = store.events[store.events.length - 1];
    return e ? e.id : undefined;
  },
  () => {
    scrollToLatestUserMessage();
  }
);

watch(() => store.streamingReply, () => {
  // Don't re-scroll on every streaming token — it would fight the user's scrolling
});

watch(() => store.pendingUserMessage, (msg) => {
  if (msg) scrollToLatestUserMessage();
});

// Reset voice cache when session changes (new session)
watch(() => store.sessionId, () => {
  sessionVoiceId = null;
});

onMounted(async () => {
  initVoice();

  if (!store.sessionId) {
    await store.startSession();
  } else {
    store.openStreamSocket();
    await store.refreshSessionState();
  }

  {
    const last = store.events[store.events.length - 1];
    lastSpokenEventId = last ? last.id : null;
  }
});

onBeforeUnmount(() => {
  stopConvai();
  void stopVoiceInput(false);
  stopSpeakingAnimation();
  if (currentAudio) {
    currentAudio.pause();
    if (currentAudio.src) URL.revokeObjectURL(currentAudio.src);
    currentAudio = null;
  }
});
</script>

<template>
  <main class="page">
    <!-- ===== User-facing screen ===== -->
    <section class="card chat-card">
      <AudioAura3D
        :level="auraLevel"
        :height="auraHeight"
        :is-listening="isRecording"
        :is-speaking="isSpeaking"
        :is-thinking="store.isThinking"
        :is-talking="store.isTalking"
      />

      <div v-if="store.session" class="status-bar">
        <span class="status-pill">{{ store.session.stage.mode.replaceAll('_', ' ') }}</span>
        <span class="status-pill">Day {{ store.session.timeline.currentDay }}/{{ store.session.timeline.totalDays }}</span>
        <span class="status-pill" :class="'health-' + store.session.health.status">{{ store.session.health.status }} {{ store.session.health.value }}</span>
        <span class="status-pill">Trust {{ store.session.bond.trust.toFixed(2) }}</span>
        <button type="button" class="status-pill new-session-btn" @click="store.newSession()" :disabled="store.loading">New Session</button>
      </div>

      <!-- Chat thread — only bubbles scroll -->
      <div class="chat-thread" ref="chatScroll" @scroll="onChatScroll">
        <div v-if="!store.events.length && !store.pendingUserMessage" class="chat-empty">
          <p>Say hello to your companion!</p>
        </div>

        <template v-for="evt in store.events" :key="evt.id">
          <div class="chat-bubble chat-user">
            <p>{{ evt.userMessage }}</p>
            <span class="chat-time">{{ new Date(evt.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}</span>
          </div>
          <div class="chat-bubble chat-ai">
            <p>{{ evt.responseText }}</p>
          </div>
        </template>

        <!-- User message shown immediately on send, before AI responds -->
        <div v-if="store.pendingUserMessage" ref="pendingBubble" class="chat-bubble chat-user">
          <p>{{ store.pendingUserMessage }}</p>
          <span class="chat-time">now</span>
        </div>

        <!-- Streaming in-progress -->
        <div v-if="store.streamingReply" class="chat-bubble chat-ai chat-streaming">
          <p>{{ store.streamingReply }}<span class="typing-cursor">|</span></p>
        </div>

        <!-- Thinking indicator -->
        <div v-else-if="store.isThinking" class="chat-bubble chat-ai chat-thinking">
          <span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
      </div>

      <!-- Composer -->
      <form class="composer" @submit.prevent="sendCurrentText">
        <div class="composer-row">
          <!-- Mic toggle inside input area -->
          <button
            type="button"
            class="icon-btn mic-toggle"
            :class="{ active: micOn }"
            :disabled="!voiceSupported"
            @click="toggleMic"
            :aria-label="micOn ? 'Turn off microphone' : 'Turn on microphone'"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" :stroke="micOn ? '#e74c3c' : 'currentColor'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          </button>
          <textarea
            v-model="store.messageDraft"
            :placeholder="micOn && isRecording ? 'Listening...' : 'Type a message...'"
            rows="1"
            @keydown.enter.exact.prevent="sendCurrentText"
          />
          <button type="submit" class="send-btn" :disabled="!store.messageDraft.trim() && !isRecording" aria-label="Send">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
          </button>
        </div>
        <div class="composer-controls">
          <div class="composer-secondary">
            <!-- Speaker toggle: AI voice on/off -->
            <button
              type="button"
              class="icon-btn speaker-toggle"
              :class="{ active: speakerOn }"
              @click="toggleSpeaker"
              :aria-label="speakerOn ? 'Mute AI voice' : 'Unmute AI voice'"
            >
              <svg v-if="speakerOn" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
              </svg>
              <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <line x1="23" y1="9" x2="17" y2="15"/>
                <line x1="17" y1="9" x2="23" y2="15"/>
              </svg>
            </button>
            <span class="voice-label">{{ speakerOn ? (convaiActive ? 'Voice on' : 'Connecting...') : 'Voice off' }}</span>
            <button type="button" @click="stopCurrentOutput" :disabled="!store.loading && !isSpeaking && !convaiActive" class="stop-btn">Stop</button>
          </div>
        </div>
      </form>

      <!-- Sticky composer bottom area for errors/transcript -->
      <div v-if="voiceError || store.error || (micOn && (transcriptFinal || transcriptInterim))" class="composer-notices">
        <p v-if="micOn && (transcriptFinal || transcriptInterim)" class="transcript">
          {{ [transcriptFinal, transcriptInterim].filter(Boolean).join(' ') }}
        </p>
        <p v-if="voiceError" class="error">{{ voiceError }}</p>
        <p v-if="store.error" class="error">{{ store.error }}</p>
      </div>
    </section>

    <!-- ===== Debug / Analytics panel ===== -->
    <details class="debug-panel" :open="debugOpen || undefined" @toggle="debugOpen = ($event.target as HTMLDetailsElement).open">
      <summary class="debug-toggle">Developer &amp; Analytics</summary>

      <section class="card debug-card">
        <div class="debug-section">
          <h3>Session State</h3>
          <div v-if="store.session" class="status-grid">
            <p><strong>Act:</strong> {{ store.session.narrative.act }}</p>
            <p><strong>Stage:</strong> {{ store.session.stage.mode }}</p>
            <p><strong>Day:</strong> {{ store.session.timeline.currentDay }} / {{ store.session.timeline.totalDays }}</p>
            <p><strong>Interactions Today:</strong> {{ store.session.timeline.interactionsToday }}</p>
            <p><strong>Health:</strong> {{ store.session.health.status }} ({{ store.session.health.value }})</p>
            <p><strong>Trust:</strong> {{ store.session.bond.trust.toFixed(2) }}</p>
            <p><strong>Maturity:</strong> {{ store.session.progress.maturity.toFixed(2) }}</p>
          </div>
        </div>

        <div class="debug-section">
          <h3>Persona</h3>
          <p>
            <strong>Active:</strong>
            <span>{{ store.activePersonaId ?? 'unknown' }}</span>
            <span v-if="store.personaCycling"> · generating…</span>
          </p>
          <div class="actions">
            <button type="button" @click="store.cyclePersona" :disabled="store.personaCycling">Cycle Persona</button>
            <button type="button" @click="store.refreshActivePersona" :disabled="store.personaCycling">Refresh</button>
            <button type="button" @click="store.startSession({ createPersona: true })" :disabled="store.loading || store.personaCycling">
              New Session + New Persona
            </button>
          </div>
          <p v-if="store.personaCycleError" class="error">{{ store.personaCycleError }}</p>
          <details v-if="store.activePersona">
            <summary>View quick summary</summary>
            <p style="white-space: pre-wrap">{{ store.activePersona.quickSummary }}</p>
          </details>
        </div>

        <div class="debug-section" v-if="appEnv.devTools">
          <h3>User Model</h3>
          <div class="actions">
            <input v-model="store.devUserId" type="text" placeholder="dev user id" maxlength="64" />
            <button type="button" @click="store.refreshUserModel" :disabled="store.userModelUpdating">Load</button>
            <button type="button" @click="store.updateUserModelFromSession" :disabled="store.userModelUpdating || !store.sessionId">
              Update From Session
            </button>
          </div>
          <p v-if="store.userModelUpdating">Updating…</p>
          <p v-if="store.userModelError" class="error">{{ store.userModelError }}</p>
          <p v-else-if="store.userModelDoc">
            <strong>Last updated:</strong>
            {{ new Date(store.userModelDoc.updated_at).toLocaleString() }}
          </p>
          <details v-if="store.userModelDoc">
            <summary>View user_character JSON</summary>
            <pre style="white-space: pre-wrap; overflow: auto; max-height: 320px">{{ JSON.stringify(store.userModelDoc.user_character, null, 2) }}</pre>
          </details>
        </div>

        <div class="debug-section" v-if="store.session">
          <h3>Mutual Knowledge</h3>
          <p>
            <strong>Points:</strong> {{ store.session.knowledge.points }} ·
            <strong>User facts:</strong> {{ store.session.knowledge.userFacts.length }} ·
            <strong>Character facets:</strong> {{ store.session.knowledge.characterFacetsRevealed.length }}
          </p>
          <div class="knowledge-grid">
            <div>
              <p><strong>Recent user facts</strong></p>
              <ul class="knowledge-list">
                <li v-for="fact in store.session.knowledge.userFacts.slice(-5)" :key="fact.id">
                  {{ fact.category }}: {{ fact.fact }}
                </li>
              </ul>
            </div>
            <div>
              <p><strong>Character facets revealed</strong></p>
              <ul class="knowledge-list">
                <li v-for="reveal in store.session.knowledge.characterFacetsRevealed.slice(-5)" :key="reveal.id">
                  {{ reveal.facet }}
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div class="debug-section" v-if="store.totalLatencyMs !== null">
          <h3>Performance</h3>
          <div class="latency-row">
            <p><strong>Total latency:</strong> {{ store.totalLatencyMs }}ms</p>
            <p><strong>Model latency:</strong> {{ store.modelLatencyMs ?? 0 }}ms</p>
            <p><strong>Path:</strong> {{ store.usedFallback ? 'fallback' : 'provider' }}</p>
          </div>
          <div class="transport-row">
            <p><strong>Streaming:</strong> {{ store.wsReady ? 'websocket' : 'http fallback' }}</p>
            <p><strong>Mic:</strong> {{ isRecording ? 'live' : 'idle' }} ({{ micLevel.toFixed(2) }})</p>
            <p><strong>Speech:</strong> {{ isSpeaking ? 'playing' : 'idle' }}</p>
          </div>
        </div>

        <div class="debug-section">
          <h3>Needs</h3>
          <div class="needs" v-if="store.needs.length">
            <div class="need" v-for="n in store.needs" :key="n.key">
              <span>{{ n.key }}</span>
              <progress :value="n.value" max="1"></progress>
              <span>{{ n.value.toFixed(2) }}</span>
            </div>
          </div>
        </div>

        <div class="debug-section">
          <h3>Progression</h3>
          <div class="actions">
            <button type="button" @click="store.tickProgression" :disabled="store.loading">Advance Day (Tick)</button>
          </div>
        </div>

        <div class="debug-section" v-if="store.outreachNudge">
          <h3>Companion Check-In</h3>
          <p>
            <strong>Severity:</strong> {{ store.outreachNudge.severity }} ·
            <strong>Inactivity:</strong> {{ store.outreachNudge.inactivityHours.toFixed(1) }}h
          </p>
          <p>{{ store.outreachNudge.message }}</p>
          <div class="actions" v-if="store.outreachNudge.askForContact">
            <input
              v-model="store.outreachContactDraft"
              type="text"
              placeholder="Email or push hint (optional)"
              maxlength="160"
            />
            <button type="button" @click="allowOutreachContact">Allow reminders</button>
            <button type="button" @click="declineOutreachContact">No reminders</button>
          </div>
          <div class="actions" v-else-if="store.outreachNudge.shouldNotify">
            <button type="button" @click="acknowledgeOutreach">Acknowledge reminder</button>
          </div>
        </div>

        <div class="debug-section" v-if="store.seedInteractions.length">
          <h3>Seed Interaction Queue</h3>
          <div class="event" v-for="seed in store.seedInteractions" :key="seed.id">
            <p class="event-top">
              <span><strong>{{ seed.stage }}</strong> · {{ seed.kind }}</span>
              <span>#{{ seed.id }}</span>
            </p>
            <p>{{ seed.prompt }}</p>
            <p class="event-meta">
              measures={{ seed.measures.join(', ') }} · tags={{ seed.tags.join(', ') }}
            </p>
          </div>
        </div>

        <div class="debug-section" v-if="store.runReport">
          <h3>Run Report (v0)</h3>
          <p><strong>Outcome:</strong> {{ store.runReport.outcome }}</p>
          <p>{{ store.runReport.summary }}</p>
          <p><strong>Strengths:</strong> {{ store.runReport.strengths.join(' | ') }}</p>
          <p><strong>Patterns:</strong> {{ store.runReport.patterns.join(' | ') }}</p>
          <p><strong>Confidence:</strong> {{ store.runReport.confidenceNotes.join(' | ') }}</p>
          <div class="event" v-for="dimension in reportDimensions" :key="dimension.key">
            <p class="event-top">
              <span><strong>{{ formatDimensionKey(dimension.key) }}</strong></span>
              <span>score {{ dimension.score.toFixed(2) }} · conf {{ dimension.confidence.toFixed(2) }}</span>
            </p>
            <p class="event-meta">{{ dimension.interpretation }}</p>
          </div>
        </div>

        <div class="debug-section" v-if="store.recentEvents.length">
          <h3>Interaction Log</h3>
          <div class="event" v-for="evt in store.recentEvents" :key="evt.id">
            <p class="event-top">
              <span><strong>{{ evt.provider }}</strong> · {{ evt.model }}</span>
              <span>{{ new Date(evt.at).toLocaleTimeString() }}</span>
            </p>
            <p><strong>User:</strong> {{ evt.userMessage }}</p>
            <p><strong>AI:</strong> {{ evt.responseText }}</p>
            <p class="event-meta">tone={{ evt.tone }} · {{ deltaSummary(evt.needDeltas) }}</p>
          </div>
        </div>
      </section>
    </details>
  </main>
</template>
