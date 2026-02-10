import type {
  InteractionEvent,
  NeedKey,
  OutreachConsent,
  OutreachNudge,
  SeedInteraction,
  SessionRunReport,
  SessionState
} from '@social-pet/domain';
import { defineStore } from 'pinia';
import { computed, ref } from 'vue';

import { appEnv } from '../../../core/config/env';

const defaultNeedsOrder: NeedKey[] = ['connection', 'safety', 'approval', 'empathy', 'autonomy', 'fun'];

type InteractionResponse = {
  state: SessionState;
  responseText: string;
  event: InteractionEvent;
  meta: {
    totalLatencyMs: number;
    modelLatencyMs: number;
    usedFallback: boolean;
  };
};

type StreamOutMessage =
  | { type: 'ready' }
  | { type: 'pong' }
  | { type: 'stream_started'; at: string }
  | { type: 'stream_stopped'; reason: 'client_stop' | 'socket_closed' | 'new_message' | 'idle' }
  | { type: 'assistant_delta'; delta: string; text: string }
  | { type: 'assistant_done'; payload: InteractionResponse }
  | { type: 'error'; error: string };

type PersonaSummary = {
  personaId: string;
  generatedAt: string;
  promptSummary: string;
  quickSummary: string;
  gender?: string;
};

type PersonaJobRecord = {
  id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  error?: string;
  result?: {
    personaId: string;
    docPath: string;
    indexPath: string;
    validationIssueCount: number;
  };
};

type UserModelDoc = {
  schema_name: string;
  schema_version: string;
  user_id: string;
  updated_at: string;
  user_character: Record<string, unknown>;
};

type UserModelJobRecord = {
  id: string;
  userId: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  error?: string;
  result?: { updatedAt: string };
};

function toWsBaseUrl(httpBaseUrl: string): string {
  if (httpBaseUrl.startsWith('https://')) {
    return httpBaseUrl.replace('https://', 'wss://');
  }
  if (httpBaseUrl.startsWith('http://')) {
    return httpBaseUrl.replace('http://', 'ws://');
  }
  return httpBaseUrl;
}

const SESSION_STORAGE_KEY = 'sp_session_id';

function loadSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
}

function saveSessionId(id: string | null): void {
  try {
    if (id) {
      localStorage.setItem(SESSION_STORAGE_KEY, id);
    } else {
      localStorage.removeItem(SESSION_STORAGE_KEY);
    }
  } catch {
    // ignore
  }
}

export const useSessionStore = defineStore('session', () => {
  const sessionId = ref<string | null>(loadSessionId());
  const session = ref<SessionState | null>(null);
  const events = ref<InteractionEvent[]>([]);
  const messageDraft = ref('');
  const loading = ref(false);
  const error = ref<string | null>(null);
  const totalLatencyMs = ref<number | null>(null);
  const modelLatencyMs = ref<number | null>(null);
  const usedFallback = ref(false);
  const streamingReply = ref('');
  const wsReady = ref(false);
  const seedInteractions = ref<SeedInteraction[]>([]);
  const runReport = ref<SessionRunReport | null>(null);
  const outreachNudge = ref<OutreachNudge | null>(null);
  const outreachContactDraft = ref('');
  const pendingUserMessage = ref<string | null>(null);
  const activePersona = ref<PersonaSummary | null>(null);
  const personaCycling = ref(false);
  const personaCycleError = ref<string | null>(null);
  function initialDevUserId(): string {
    try {
      return localStorage.getItem('sp_dev_user_id') ?? 'dev_user';
    } catch {
      return 'dev_user';
    }
  }

  const devUserId = ref<string>(initialDevUserId());
  const userModelDoc = ref<UserModelDoc | null>(null);
  const userModelUpdating = ref(false);
  const userModelError = ref<string | null>(null);

  function setDevUserId(next: string): void {
    devUserId.value = next;
    try {
      localStorage.setItem('sp_dev_user_id', next);
    } catch {
      // ignore
    }
  }

  // Fires as soon as full response text is available (before reveal animation)
  // so TTS can start in parallel with the character-by-character text reveal.
  const pendingTtsText = ref<{ id: string; text: string } | null>(null);

  let ws: WebSocket | null = null;
  let httpRequestController: AbortController | null = null;

  // --- Speaking-pace token reveal ---
  // Buffers fast-arriving LLM tokens and reveals them character-by-character
  // at a natural speaking pace so the talking animation has time to play.
  let _revealBuffer = '';
  let _revealPos = 0;
  let _revealTimer: ReturnType<typeof setTimeout> | null = null;
  let _revealDonePayload: InteractionResponse | null = null;

  const REVEAL_CHAR_MS = 35;
  const REVEAL_SPACE_MS = 45;
  const REVEAL_COMMA_MS = 100;
  const REVEAL_SENTENCE_MS = 180;

  function clearRevealState(): void {
    if (_revealTimer !== null) {
      clearTimeout(_revealTimer);
      _revealTimer = null;
    }
    _revealBuffer = '';
    _revealPos = 0;
    _revealDonePayload = null;
  }

  function flushReveal(): void {
    if (!_revealDonePayload) return;
    const payload = _revealDonePayload;
    clearRevealState();
    streamingReply.value = '';
    applyInteractionResponse(payload);
    loading.value = false;
    void refreshSeedInteractions();
    void refreshRunReport();
    void refreshOutreachNudge();
  }

  function tickReveal(): void {
    _revealTimer = null;

    if (_revealPos >= _revealBuffer.length) {
      if (_revealDonePayload) flushReveal();
      return;
    }

    _revealPos += 1;
    streamingReply.value = _revealBuffer.slice(0, _revealPos);

    const justRevealed = _revealBuffer[_revealPos - 1];
    let delay = REVEAL_CHAR_MS;
    if (justRevealed === '.' || justRevealed === '!' || justRevealed === '?') {
      delay = REVEAL_SENTENCE_MS;
    } else if (justRevealed === ',') {
      delay = REVEAL_COMMA_MS;
    } else if (justRevealed === ' ') {
      delay = REVEAL_SPACE_MS;
    }

    _revealTimer = setTimeout(tickReveal, delay);
  }

  function feedRevealBuffer(text: string): void {
    _revealBuffer = text;
    if (_revealTimer === null && _revealPos < _revealBuffer.length) {
      tickReveal();
    }
  }

  function finishReveal(payload: InteractionResponse): void {
    _revealDonePayload = payload;
    if (_revealPos >= _revealBuffer.length) {
      flushReveal();
    }
  }

  const needs = computed(() => {
    if (!session.value) return [];
    return defaultNeedsOrder.map((key) => ({
      key,
      value: session.value!.needs[key]
    }));
  });

  const recentEvents = computed(() => events.value.slice().reverse());
  const creatureLine = computed(() => streamingReply.value || session.value?.latestResponseText || '');
  const isTalking = computed(() => streamingReply.value.length > 0);
  const isThinking = computed(() => loading.value && !isTalking.value);
  const activePersonaId = computed(() => activePersona.value?.personaId ?? null);

  function applyInteractionResponse(data: InteractionResponse): void {
    session.value = data.state;
    events.value = [...events.value, data.event];
    pendingUserMessage.value = null;
    totalLatencyMs.value = data.meta.totalLatencyMs;
    modelLatencyMs.value = data.meta.modelLatencyMs;
    usedFallback.value = data.meta.usedFallback;
  }

  async function refreshActivePersona(): Promise<void> {
    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/persona/active`);
      if (!res.ok) return;
      const data = (await res.json()) as { summary: PersonaSummary | null };
      activePersona.value = data.summary;
    } catch {
      // ignore (persona endpoints might not be enabled)
    }
  }

  async function cyclePersona(): Promise<void> {
    personaCycling.value = true;
    personaCycleError.value = null;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/persona/cycle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (!res.ok) throw new Error(`persona cycle failed (${res.status})`);

      const data = (await res.json()) as { jobId: string };
      const jobId = data.jobId;

      const startedAt = Date.now();
      while (Date.now() - startedAt < 240_000) {
        const jobRes = await fetch(`${appEnv.apiBaseUrl}/persona/jobs/${jobId}`);
        if (!jobRes.ok) throw new Error(`persona job status failed (${jobRes.status})`);
        const job = (await jobRes.json()) as PersonaJobRecord;

        if (job.status === 'succeeded') {
          await refreshActivePersona();
          return;
        }
        if (job.status === 'failed') {
          throw new Error(job.error ?? 'persona_job_failed');
        }

        await new Promise((r) => setTimeout(r, 1200));
      }

      throw new Error('persona_job_timeout');
    } catch (err) {
      personaCycleError.value = err instanceof Error ? err.message : 'persona_cycle_failed';
    } finally {
      personaCycling.value = false;
    }
  }

  function openStreamSocket(): void {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    const wsUrl = `${toWsBaseUrl(appEnv.apiBaseUrl)}/interaction/stream/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsReady.value = true;
    };

    ws.onclose = () => {
      wsReady.value = false;
      ws = null;
    };

    ws.onerror = () => {
      wsReady.value = false;
    };

    ws.onmessage = (event) => {
      let msg: StreamOutMessage;

      try {
        msg = JSON.parse(event.data) as StreamOutMessage;
      } catch {
        return;
      }

      if (msg.type === 'assistant_delta') {
        feedRevealBuffer(msg.text);
        return;
      }

      if (msg.type === 'stream_started') {
        loading.value = true;
        clearRevealState();
        streamingReply.value = '';
        return;
      }

      if (msg.type === 'assistant_done') {
        // Signal TTS immediately — don't wait for reveal animation to finish
        pendingTtsText.value = {
          id: msg.payload.event.id,
          text: msg.payload.event.responseText
        };
        finishReveal(msg.payload);
        return;
      }

      if (msg.type === 'stream_stopped') {
        clearRevealState();
        streamingReply.value = '';
        loading.value = false;
        return;
      }

      if (msg.type === 'error') {
        error.value = msg.error;
        clearRevealState();
        streamingReply.value = '';
        loading.value = false;
      }
    };
  }

  async function refreshUserModel(): Promise<void> {
    if (!appEnv.devTools) return;
    if (!devUserId.value.trim()) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/user/${encodeURIComponent(devUserId.value.trim())}/model`);
      if (!res.ok) return;
      userModelDoc.value = (await res.json()) as UserModelDoc;
    } catch {
      // ignore
    }
  }

  async function updateUserModelFromSession(): Promise<void> {
    if (!appEnv.devTools) return;
    if (!devUserId.value.trim()) return;
    if (!sessionId.value) return;

    userModelUpdating.value = true;
    userModelError.value = null;

    try {
      const userId = devUserId.value.trim();
      const res = await fetch(`${appEnv.apiBaseUrl}/user/${encodeURIComponent(userId)}/model/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: sessionId.value })
      });
      if (!res.ok) throw new Error(`user model update failed (${res.status})`);

      const data = (await res.json()) as { jobId: string };
      const jobId = data.jobId;

      const startedAt = Date.now();
      while (Date.now() - startedAt < 240_000) {
        const jobRes = await fetch(`${appEnv.apiBaseUrl}/user/${encodeURIComponent(userId)}/model/jobs/${jobId}`);
        if (!jobRes.ok) throw new Error(`user model job status failed (${jobRes.status})`);
        const job = (await jobRes.json()) as UserModelJobRecord;

        if (job.status === 'succeeded') {
          await refreshUserModel();
          return;
        }
        if (job.status === 'failed') {
          throw new Error(job.error ?? 'user_model_job_failed');
        }

        await new Promise((r) => setTimeout(r, 1200));
      }

      throw new Error('user_model_job_timeout');
    } catch (err) {
      userModelError.value = err instanceof Error ? err.message : 'user_model_update_failed';
    } finally {
      userModelUpdating.value = false;
    }
  }

  async function startSession(options?: { createPersona?: boolean }) {
    loading.value = true;
    error.value = null;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...(options?.createPersona ? { createPersona: true } : {}),
          ...(appEnv.devTools && devUserId.value.trim().length > 0 ? { userId: devUserId.value.trim() } : {})
        })
      });

      if (!res.ok) throw new Error(`start session failed (${res.status})`);

      const data = (await res.json()) as { sessionId: string; state: SessionState; events: InteractionEvent[] };
      sessionId.value = data.sessionId;
      saveSessionId(data.sessionId);
      session.value = data.state;
      events.value = data.events;

      openStreamSocket();
      void refreshSeedInteractions();
      void refreshRunReport();
      void refreshOutreachNudge();
      void refreshActivePersona();
      void refreshUserModel();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    } finally {
      loading.value = false;
    }
  }

  async function refreshSessionState() {
    if (!sessionId.value) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/${sessionId.value}/state`);
      if (!res.ok) throw new Error(`state fetch failed (${res.status})`);

      const data = (await res.json()) as { state: SessionState; events: InteractionEvent[] };
      session.value = data.state;
      events.value = data.events;
      void refreshSeedInteractions();
      void refreshRunReport();
      void refreshOutreachNudge();
      void refreshActivePersona();
      void refreshUserModel();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  async function sendMessage(messageOverride?: string) {
    if (!sessionId.value) return;

    const outgoing = (messageOverride ?? messageDraft.value).trim();
    if (!outgoing) return;

    error.value = null;
    loading.value = true;
    clearRevealState();
    streamingReply.value = '';
    pendingUserMessage.value = outgoing;

    if (wsReady.value && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'user_message',
          sessionId: sessionId.value,
          message: outgoing
        })
      );
      if (!messageOverride) messageDraft.value = '';
      return;
    }

    const requestController = new AbortController();
    httpRequestController = requestController;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/interaction/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: requestController.signal,
        body: JSON.stringify({
          sessionId: sessionId.value,
          message: outgoing
        })
      });

      if (!res.ok) throw new Error(`interaction failed (${res.status})`);

      const data = (await res.json()) as InteractionResponse;
      pendingTtsText.value = { id: data.event.id, text: data.event.responseText };
      applyInteractionResponse(data);
      if (!messageOverride) messageDraft.value = '';
      void refreshSeedInteractions();
      void refreshRunReport();
      void refreshOutreachNudge();
    } catch (err) {
      if (requestController.signal.aborted) return;
      error.value = err instanceof Error ? err.message : 'Unknown error';
    } finally {
      if (httpRequestController === requestController) {
        httpRequestController = null;
      }
      loading.value = false;
      streamingReply.value = '';
    }
  }

  function stopCurrentStream(): void {
    if (wsReady.value && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop_stream' }));
    }

    if (httpRequestController) {
      httpRequestController.abort(new Error('stopped_by_client'));
      httpRequestController = null;
    }

    clearRevealState();
    loading.value = false;
    streamingReply.value = '';
    pendingUserMessage.value = null;
  }

  async function tickProgression() {
    if (!sessionId.value) return;

    loading.value = true;
    error.value = null;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/progression/tick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: sessionId.value })
      });

      if (!res.ok) throw new Error(`tick failed (${res.status})`);

      const data = (await res.json()) as { state: SessionState };
      session.value = data.state;
      void refreshSeedInteractions();
      void refreshRunReport();
      void refreshOutreachNudge();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    } finally {
      loading.value = false;
    }
  }

  async function refreshSeedInteractions(limit = 8): Promise<void> {
    if (!sessionId.value) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/${sessionId.value}/seed-interactions?limit=${limit}`);
      if (!res.ok) throw new Error(`seed interactions failed (${res.status})`);

      const data = (await res.json()) as { interactions: SeedInteraction[] };
      seedInteractions.value = data.interactions;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  async function refreshRunReport(): Promise<void> {
    if (!sessionId.value) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/${sessionId.value}/report`);
      if (!res.ok) throw new Error(`run report failed (${res.status})`);

      runReport.value = (await res.json()) as SessionRunReport;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  async function refreshOutreachNudge(): Promise<void> {
    if (!sessionId.value) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/${sessionId.value}/outreach`);
      if (!res.ok) throw new Error(`outreach failed (${res.status})`);

      outreachNudge.value = (await res.json()) as OutreachNudge;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  async function setOutreachPreference(consent: OutreachConsent, contactHint?: string): Promise<void> {
    if (!sessionId.value) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/${sessionId.value}/outreach/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          consent,
          contactHint: contactHint?.trim() ? contactHint.trim() : undefined
        })
      });

      if (!res.ok) throw new Error(`outreach preference failed (${res.status})`);

      const data = (await res.json()) as { state: SessionState };
      session.value = data.state;
      await refreshOutreachNudge();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  async function markOutreachSent(): Promise<void> {
    if (!sessionId.value) return;

    try {
      const res = await fetch(`${appEnv.apiBaseUrl}/session/${sessionId.value}/outreach/mark-sent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (!res.ok) throw new Error(`outreach mark-sent failed (${res.status})`);

      const data = (await res.json()) as { state: SessionState };
      session.value = data.state;
      await refreshOutreachNudge();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  async function newSession(): Promise<void> {
    // Close existing connections
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close();
      ws = null;
      wsReady.value = false;
    }

    // Clear state
    sessionId.value = null;
    saveSessionId(null);
    session.value = null;
    events.value = [];
    streamingReply.value = '';
    pendingUserMessage.value = null;
    pendingTtsText.value = null;
    error.value = null;
    seedInteractions.value = [];
    runReport.value = null;
    outreachNudge.value = null;

    // Start fresh
    await startSession();
  }

  return {
    sessionId,
    session,
    events,
    recentEvents,
    messageDraft,
    loading,
    error,
    totalLatencyMs,
    modelLatencyMs,
    usedFallback,
    streamingReply,
    wsReady,
    seedInteractions,
    runReport,
    outreachNudge,
    outreachContactDraft,
    pendingUserMessage,
    pendingTtsText,
    creatureLine,
    isTalking,
    isThinking,
    activePersona,
    activePersonaId,
    personaCycling,
    personaCycleError,
    devUserId,
    setDevUserId,
    userModelDoc,
    userModelUpdating,
    userModelError,
    needs,
    startSession,
    newSession,
    refreshActivePersona,
    cyclePersona,
    refreshUserModel,
    updateUserModelFromSession,
    refreshSessionState,
    sendMessage,
    stopCurrentStream,
    tickProgression,
    openStreamSocket,
    refreshSeedInteractions,
    refreshRunReport,
    refreshOutreachNudge,
    setOutreachPreference,
    markOutreachSent
  };
});
