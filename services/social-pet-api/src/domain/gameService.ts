import { randomUUID } from 'node:crypto';

import type {
  InteractionEvent,
  InteractionTone,
  NeedDeltaState,
  NeedKey,
  OutreachConsent,
  OutreachNudge,
  SeedInteraction,
  SessionRunReport,
  SessionState,
  SocialNeedState
} from '@social-pet/domain';

import {
  actForTimelineDay,
  activeTrialForTimeline,
  advanceTimelineDay,
  applyAssessmentObservation,
  applyHealthDayAdvance,
  applyHealthFromInteraction,
  applyMutualKnowledgeFromInteraction,
  buildRunReport,
  canAcceptInteraction,
  createInitialAssessmentState,
  createInitialHealthState,
  createInitialKnowledgeState,
  createInitialTimelineState,
  deriveOutcome,
  evaluateOutreachNudge,
  hoursBetween,
  registerInteractionInTimeline,
  stageForTimeline,
  wholeDaysBetween
} from './phase1Engines';
import type { ConversationTurn, LLMGateway, LLMInput, LLMOutput } from './llmGateway';
import type { SessionPersistence, SessionRecord } from './persistenceStore';
import { getSeedInteractionsForStage, pickSeedInteraction } from './seedInteractions';

const needsOrder: NeedKey[] = ['connection', 'safety', 'approval', 'empathy', 'autonomy', 'fun'];

export interface GameService {
  startSession: (opts?: { userId?: string }) => Promise<{ sessionId: string; state: SessionState; events: InteractionEvent[] }>;
  getSession: (sessionId: string) => Promise<SessionRecord | null>;
  respondToInteraction: (
    sessionId: string,
    message: string
  ) => Promise<
    | {
        state: SessionState;
        responseText: string;
        event: InteractionEvent;
        meta: { totalLatencyMs: number; modelLatencyMs: number; usedFallback: boolean };
      }
    | null
  >;
  streamInteraction: (
    sessionId: string,
    message: string,
    onToken: (delta: string, accumulated: string) => void,
    options?: { signal?: AbortSignal }
  ) => Promise<
    | {
        state: SessionState;
        responseText: string;
        event: InteractionEvent;
        meta: { totalLatencyMs: number; modelLatencyMs: number; usedFallback: boolean };
      }
    | null
  >;
  tickProgression: (sessionId: string) => Promise<SessionState | null>;
  renderAppearancePrompt: (sessionId: string) => Promise<{ prompt: string } | null>;
  getSeedInteractions: (sessionId: string, limit?: number) => Promise<SeedInteraction[] | null>;
  generateRunReport: (sessionId: string) => Promise<SessionRunReport | null>;
  getOutreachNudge: (sessionId: string) => Promise<OutreachNudge | null>;
  updateOutreachPreference: (
    sessionId: string,
    preference: { consent: OutreachConsent; contactHint?: string }
  ) => Promise<SessionState | null>;
  markOutreachSent: (sessionId: string) => Promise<SessionState | null>;
}

function clamp01(value: number): number {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function makeInitialNeeds(): SocialNeedState {
  return {
    connection: 0.55,
    safety: 0.7,
    approval: 0.5,
    empathy: 0.45,
    autonomy: 0.6,
    fun: 0.5
  };
}

function makeInitialState(sessionId: string, userId?: string): SessionState {
  const nowIso = new Date().toISOString();
  return {
    sessionId,
    userId,
    needs: makeInitialNeeds(),
    stage: { mode: 'young_adult' },
    knowledge: createInitialKnowledgeState(),
    bond: { trust: 0.5, ruptureRepairBalance: 0.5 },
    progress: { maturity: 0.05, interactions: 0 },
    narrative: {
      act: 'safe_bonding',
      activeTrial: null
    },
    timeline: createInitialTimelineState(nowIso),
    assessment: createInitialAssessmentState(),
    health: createInitialHealthState(),
    outcome: {
      ended: false
    },
    outreach: {
      consent: 'unknown',
      nudgeCount: 0
    }
  };
}

function inferTone(message: string): InteractionTone {
  const text = message.toLowerCase();
  const positive = ['love', 'proud', 'great', 'thanks', 'sorry', 'care', 'friend', 'listen'];
  const negative = ['hate', 'stupid', 'annoying', 'shut up', 'bad', 'boring'];

  const posHit = positive.some((word) => text.includes(word));
  const negHit = negative.some((word) => text.includes(word));

  if (negHit) return 'harsh';
  if (posHit) return 'supportive';
  return 'neutral';
}

function computeNeedDeltas(tone: InteractionTone): NeedDeltaState {
  if (tone === 'supportive') {
    return {
      connection: 0.08,
      safety: 0.03,
      approval: 0.04,
      empathy: 0.06,
      fun: 0.02
    };
  }

  if (tone === 'harsh') {
    return {
      safety: -0.08,
      connection: -0.06,
      approval: -0.05,
      empathy: -0.04
    };
  }

  return {
    connection: 0.01,
    autonomy: 0.01,
    fun: 0.01
  };
}

function trustDeltaForTone(tone: InteractionTone): number {
  if (tone === 'supportive') return 0.04;
  if (tone === 'harsh') return -0.06;
  return 0.005;
}

function ruptureRepairDeltaForTone(tone: InteractionTone): number {
  if (tone === 'supportive') return 0.02;
  if (tone === 'harsh') return -0.03;
  return 0;
}

function applyNeedDeltas(needs: SocialNeedState, deltas: NeedDeltaState): SocialNeedState {
  return {
    connection: clamp01(needs.connection + (deltas.connection ?? 0)),
    safety: clamp01(needs.safety + (deltas.safety ?? 0)),
    approval: clamp01(needs.approval + (deltas.approval ?? 0)),
    empathy: clamp01(needs.empathy + (deltas.empathy ?? 0)),
    autonomy: clamp01(needs.autonomy + (deltas.autonomy ?? 0)),
    fun: clamp01(needs.fun + (deltas.fun ?? 0))
  };
}

function applyNeedDecay(state: SessionState): SocialNeedState {
  return needsOrder.reduce((acc, key) => {
    acc[key] = clamp01(state.needs[key] - 0.01);
    return acc;
  }, {} as SocialNeedState);
}

function buildConversationHistory(events: InteractionEvent[], maxTurns: number): ConversationTurn[] {
  const recent = events.slice(-maxTurns);
  return recent.flatMap((evt) => [
    { role: 'user', content: evt.userMessage },
    { role: 'assistant', content: evt.responseText }
  ]);
}

function latestOpenAIResponseId(events: InteractionEvent[]): string | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const evt = events[i];
    if (evt.provider === 'openai' && evt.providerResponseId) {
      return evt.providerResponseId;
    }
  }
  return undefined;
}

function projectStateByWallTime(state: SessionState, nowIso: string): { state: SessionState; changed: boolean } {
  const daysElapsed = wholeDaysBetween(state.timeline.dayStartedAt, nowIso);
  if (daysElapsed <= 0) {
    return { state, changed: false };
  }

  let nextState = state;
  let changed = false;

  for (let i = 0; i < daysElapsed; i += 1) {
    const advancedDay = advanceTimelineDay(nextState.timeline, nowIso);
    const healthDay = applyHealthDayAdvance(nextState.health, advancedDay.hadInteractionToday);
    const nextStageMode = stageForTimeline(advancedDay.timeline.currentDay, nextState.progress.interactions);
    const nextOutcome = deriveOutcome(
      nextState.outcome,
      nowIso,
      nextStageMode,
      advancedDay.timeline,
      healthDay.health.status,
      nextState.progress.interactions
    );

    nextState = {
      ...nextState,
      needs: applyNeedDecay(nextState),
      stage: { mode: nextStageMode },
      narrative: {
        act: actForTimelineDay(advancedDay.timeline.currentDay),
        activeTrial: activeTrialForTimeline(advancedDay.timeline.currentDay, healthDay.health.status)
      },
      timeline: advancedDay.timeline,
      health: healthDay.health,
      outcome: nextOutcome
    };
    changed = true;
  }

  return { state: nextState, changed };
}

export function createGameService(
  llmGateway: LLMGateway,
  persistence: SessionPersistence,
  opts: {
    eventLogMax: number;
    historyTurns: number;
    logger?: { warn: (data: unknown, msg?: string) => void };
    personaContextProvider?: (message: string, options?: { signal?: AbortSignal }) => Promise<string | undefined>;
    transcriptContextProvider?: (params: { sessionId: string; message: string; signal?: AbortSignal }) => Promise<string | undefined>;
    onEventPersisted?: (record: {
      sessionId: string;
      state: SessionState;
      events: InteractionEvent[];
      event: InteractionEvent;
    }) => void;
  }
): GameService {
  const sessions = new Map<string, SessionRecord>();

  function migrateStageMode(mode: unknown): SessionState['stage']['mode'] {
    if (mode === 'young_adult' || mode === 'middle_aged' || mode === 'wise' || mode === 'old') return mode;
    if (mode === 'toddler_like') return 'young_adult';
    if (mode === 'child_like') return 'middle_aged';
    if (mode === 'adolescent_like') return 'wise';
    if (mode === 'adult_like') return 'old';
    return 'young_adult';
  }

  function migrateLoadedState(state: SessionState): SessionState {
    const maybe = state as unknown as Partial<SessionState> & { stage?: { mode?: unknown } };

    const migratedStageMode = migrateStageMode(maybe.stage?.mode);
    const migratedKnowledge =
      maybe.knowledge && typeof maybe.knowledge.points === 'number'
        ? maybe.knowledge
        : createInitialKnowledgeState();

    return {
      ...state,
      stage: { mode: migratedStageMode },
      knowledge: migratedKnowledge
    };
  }

  function persistAsync(work: Promise<void>, context: string): void {
    void work.catch((error) => {
      opts.logger?.warn({ error, context }, 'async persistence failure');
    });
  }

  async function hydrateSession(sessionId: string): Promise<SessionRecord | null> {
    const nowIso = new Date().toISOString();
    const cached = sessions.get(sessionId);
    const loaded = cached ?? (await persistence.load(sessionId));
    if (!loaded) return null;

    const migratedState = migrateLoadedState(loaded.state);
    const projected = projectStateByWallTime(migratedState, nowIso);
    const nextRecord: SessionRecord = {
      state: projected.state,
      events: loaded.events
    };

    sessions.set(sessionId, nextRecord);

    if (projected.changed) {
      persistAsync(persistence.upsert(sessionId, nextRecord.state, nextRecord.events), 'walltime_project_upsert');
    }

    return nextRecord;
  }

  async function runInteraction(
    sessionId: string,
    message: string,
    modelCall: (input: LLMInput, options?: { signal?: AbortSignal }) => Promise<LLMOutput>,
    options?: { signal?: AbortSignal }
  ): Promise<
    | {
        state: SessionState;
        responseText: string;
        event: InteractionEvent;
        meta: { totalLatencyMs: number; modelLatencyMs: number; usedFallback: boolean };
      }
    | null
  > {
    const startedAt = Date.now();
    const record = await hydrateSession(sessionId);
    if (!record) return null;

    const gate = canAcceptInteraction(record.state.timeline, record.state.outcome);
    if (!gate.ok) {
      throw new Error(gate.reason ?? 'interaction_blocked');
    }

    const tone = inferTone(message);
    const deltas = computeNeedDeltas(tone);
    const seedInteraction = pickSeedInteraction(
      record.state.stage.mode,
      record.state.timeline.currentDay,
      record.state.progress.interactions
    );

    let personaContext: string | undefined;
    if (opts.personaContextProvider) {
      try {
        personaContext = await opts.personaContextProvider(message, options);
      } catch (error) {
        opts.logger?.warn({ error }, 'persona context provider failed; continuing without persona context');
      }
    }

    let transcriptContext: string | undefined;
    if (opts.transcriptContextProvider) {
      try {
        transcriptContext = await opts.transcriptContextProvider({ sessionId, message, signal: options?.signal });
      } catch (error) {
        opts.logger?.warn({ error }, 'transcript context provider failed; continuing without transcript context');
      }
    }

    const combinedContext = [personaContext?.trim(), transcriptContext?.trim()].filter(Boolean).join('\n\n');

    const modelResult = await modelCall(
      {
        userMessage: message,
        state: record.state,
        tone,
        history: buildConversationHistory(record.events, opts.historyTurns),
        previousProviderResponseId: latestOpenAIResponseId(record.events),
        personaContext: combinedContext.length > 0 ? combinedContext : undefined
      },
      options
    );

    const eventId = randomUUID();
    const nowIso = new Date().toISOString();
    const nextInteractions = record.state.progress.interactions + 1;
    const nextTimeline = registerInteractionInTimeline(record.state.timeline, nowIso);
    const healthResult = applyHealthFromInteraction(record.state.health, tone);
    const assessmentResult = applyAssessmentObservation(
      record.state.assessment,
      message,
      tone,
      eventId,
      nowIso,
      seedInteraction?.measures ?? [],
      seedInteraction?.id
    );
    const nextStageMode = stageForTimeline(nextTimeline.currentDay, nextInteractions);
    const nextAct = actForTimelineDay(nextTimeline.currentDay);
    const nextOutcome = deriveOutcome(
      record.state.outcome,
      nowIso,
      nextStageMode,
      nextTimeline,
      healthResult.health.status,
      nextInteractions
    );

    const baseNextState: SessionState = {
      ...record.state,
      needs: applyNeedDeltas(record.state.needs, deltas),
      stage: { mode: nextStageMode },
      bond: {
        trust: clamp01(record.state.bond.trust + trustDeltaForTone(tone)),
        ruptureRepairBalance: clamp01(record.state.bond.ruptureRepairBalance + ruptureRepairDeltaForTone(tone))
      },
      progress: {
        interactions: nextInteractions,
        maturity: clamp01(nextInteractions / 60)
      },
      narrative: {
        act: nextAct,
        activeTrial: activeTrialForTimeline(nextTimeline.currentDay, healthResult.health.status)
      },
      timeline: nextTimeline,
      health: healthResult.health,
      assessment: assessmentResult.assessment,
      outcome: nextOutcome,
      latestResponseText: modelResult.text
    };

    const event: InteractionEvent = {
      id: eventId,
      at: nowIso,
      userMessage: message,
      responseText: modelResult.text,
      tone,
      needDeltas: deltas,
      provider: modelResult.provider,
      model: modelResult.model,
      providerResponseId: modelResult.providerResponseId,
      sessionDay: record.state.timeline.currentDay,
      seedInteractionId: seedInteraction?.id,
      assessmentScores: assessmentResult.scores,
      assessmentSignals: assessmentResult.signals,
      healthDelta: healthResult.delta,
      healthStatus: healthResult.health.status
    };

    const knowledgeResult = applyMutualKnowledgeFromInteraction(record.state.knowledge, event);
    const nextState: SessionState = {
      ...baseNextState,
      knowledge: knowledgeResult.knowledge
    };

    const nextEvents = [...record.events, event].slice(-opts.eventLogMax);
    const nextRecord: SessionRecord = {
      state: nextState,
      events: nextEvents
    };

    sessions.set(sessionId, nextRecord);

    persistAsync(persistence.upsert(sessionId, nextState, nextEvents), 'interaction_upsert');
    persistAsync(persistence.appendEvent(sessionId, event), 'interaction_event_append');
    opts.onEventPersisted?.({ sessionId, state: nextState, events: nextEvents, event });

    return {
      state: nextState,
      responseText: modelResult.text,
      event,
      meta: {
        totalLatencyMs: Date.now() - startedAt,
        modelLatencyMs: modelResult.latencyMs,
        usedFallback: modelResult.fallback
      }
    };
  }

  return {
    async startSession(startOpts) {
      const sessionId = `sp_${Math.random().toString(36).slice(2, 10)}`;
      const state = makeInitialState(sessionId, startOpts?.userId);
      const record: SessionRecord = { state, events: [] };
      sessions.set(sessionId, record);

      persistAsync(persistence.upsert(sessionId, record.state, record.events), 'start_session_upsert');

      return { sessionId, state, events: [] };
    },

    async getSession(sessionId) {
      return hydrateSession(sessionId);
    },

    async respondToInteraction(sessionId, message) {
      return runInteraction(sessionId, message, (input) => llmGateway.generateReply(input));
    },

    async streamInteraction(sessionId, message, onToken, options) {
      return runInteraction(
        sessionId,
        message,
        (input, modelOptions) => {
          return llmGateway.streamReply(input, onToken, modelOptions);
        },
        options
      );
    },

    async tickProgression(sessionId) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      const nowIso = new Date().toISOString();
      const advancedDay = advanceTimelineDay(record.state.timeline, nowIso);
      const healthDay = applyHealthDayAdvance(record.state.health, advancedDay.hadInteractionToday);
      const nextStageMode = stageForTimeline(advancedDay.timeline.currentDay, record.state.progress.interactions);
      const nextOutcome = deriveOutcome(
        record.state.outcome,
        nowIso,
        nextStageMode,
        advancedDay.timeline,
        healthDay.health.status,
        record.state.progress.interactions
      );

      const nextState: SessionState = {
        ...record.state,
        needs: applyNeedDecay(record.state),
        stage: { mode: nextStageMode },
        narrative: {
          act: actForTimelineDay(advancedDay.timeline.currentDay),
          activeTrial: activeTrialForTimeline(advancedDay.timeline.currentDay, healthDay.health.status)
        },
        timeline: advancedDay.timeline,
        health: healthDay.health,
        outcome: nextOutcome
      };

      const nextRecord: SessionRecord = {
        state: nextState,
        events: record.events
      };

      sessions.set(sessionId, nextRecord);
      persistAsync(persistence.upsert(sessionId, nextState, record.events), 'tick_upsert');

      return nextState;
    },

    async renderAppearancePrompt(sessionId) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      const { stage, narrative, bond, health } = record.state;
      const prompt = [
        'friendly companion portrait, stylized creature character',
        `life phase: ${stage.mode}`,
        `hero journey act: ${narrative.act}`,
        `trust signal: ${bond.trust.toFixed(2)}`,
        `health aura: ${health.status}`,
        'expressive eyes, cohesive silhouette, cozy mature tone'
      ].join(', ');

      return { prompt };
    },

    async getSeedInteractions(sessionId, limit = 8) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      const choices = getSeedInteractionsForStage(record.state.stage.mode, record.state.timeline.currentDay);
      return choices.slice(0, Math.max(1, Math.min(limit, 12)));
    },

    async generateRunReport(sessionId) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      return buildRunReport(
        sessionId,
        {
          stageMode: record.state.stage.mode,
          currentDay: record.state.timeline.currentDay,
          healthStatus: record.state.health.status,
          healthValue: record.state.health.value,
          outcome: record.state.outcome,
          assessment: record.state.assessment,
          knowledge: record.state.knowledge
        },
        record.events
      );
    },

    async getOutreachNudge(sessionId) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      const nudge = evaluateOutreachNudge({
        lastInteractionAt: record.state.timeline.lastInteractionAt,
        dayStartedAt: record.state.timeline.dayStartedAt,
        healthStatus: record.state.health.status,
        healthValue: record.state.health.value,
        consent: record.state.outreach.consent,
        outcomeEnded: record.state.outcome.ended
      });

      if (!nudge.shouldNotify || !record.state.outreach.lastNudgeAt) {
        return nudge;
      }

      const hoursSinceLastNudge = hoursBetween(record.state.outreach.lastNudgeAt, new Date().toISOString());
      if (hoursSinceLastNudge < 6) {
        return {
          ...nudge,
          shouldNotify: false,
          severity: 'none',
          message: 'Recent reminder already sent.'
        };
      }

      return nudge;
    },

    async updateOutreachPreference(sessionId, preference) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      const nextState: SessionState = {
        ...record.state,
        outreach: {
          ...record.state.outreach,
          consent: preference.consent,
          contactHint: preference.contactHint?.trim() ? preference.contactHint.trim() : record.state.outreach.contactHint
        }
      };
      const nextRecord: SessionRecord = {
        state: nextState,
        events: record.events
      };

      sessions.set(sessionId, nextRecord);
      persistAsync(persistence.upsert(sessionId, nextState, record.events), 'outreach_preference_upsert');
      return nextState;
    },

    async markOutreachSent(sessionId) {
      const record = await hydrateSession(sessionId);
      if (!record) return null;

      const nowIso = new Date().toISOString();
      const nextState: SessionState = {
        ...record.state,
        outreach: {
          ...record.state.outreach,
          lastNudgeAt: nowIso,
          nudgeCount: record.state.outreach.nudgeCount + 1
        }
      };
      const nextRecord: SessionRecord = {
        state: nextState,
        events: record.events
      };

      sessions.set(sessionId, nextRecord);
      persistAsync(persistence.upsert(sessionId, nextState, record.events), 'outreach_mark_sent_upsert');
      return nextState;
    }
  };
}
