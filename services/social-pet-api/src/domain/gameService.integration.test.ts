import assert from 'node:assert/strict';
import test from 'node:test';

import type { InteractionEvent, SessionState } from '@social-pet/domain';

import { createGameService } from './gameService';
import type { LLMGateway, LLMInput, LLMOutput } from './llmGateway';
import {
  createInitialAssessmentState,
  createInitialHealthState,
  createInitialKnowledgeState,
  createInitialTimelineState
} from './phase1Engines';
import type { SessionPersistence, SessionRecord } from './persistenceStore';

class InMemoryPersistence implements SessionPersistence {
  readonly records = new Map<string, SessionRecord>();
  loadCalls = 0;
  upsertCalls = 0;
  appendEventCalls = 0;

  async init(): Promise<void> {
    // no-op
  }

  async load(sessionId: string): Promise<SessionRecord | null> {
    this.loadCalls += 1;
    const record = this.records.get(sessionId);
    return record ? structuredClone(record) : null;
  }

  async upsert(sessionId: string, state: SessionState, events: InteractionEvent[]): Promise<void> {
    this.upsertCalls += 1;
    this.records.set(sessionId, {
      state: structuredClone(state),
      events: structuredClone(events)
    });
  }

  async appendEvent(_sessionId: string, _event: InteractionEvent): Promise<void> {
    this.appendEventCalls += 1;
  }
}

function makeState(sessionId: string, dayStartedAt: string): SessionState {
  const timeline = createInitialTimelineState(dayStartedAt);
  return {
    sessionId,
    needs: {
      connection: 0.55,
      safety: 0.7,
      approval: 0.5,
      empathy: 0.45,
      autonomy: 0.6,
      fun: 0.5
    },
    stage: { mode: 'young_adult' },
    knowledge: createInitialKnowledgeState(),
    bond: { trust: 0.5, ruptureRepairBalance: 0.5 },
    progress: { maturity: 0.05, interactions: 0 },
    narrative: { act: 'safe_bonding', activeTrial: null },
    timeline: {
      ...timeline,
      startedAt: dayStartedAt,
      dayStartedAt,
      currentDay: 1,
      interactionsToday: 0,
      sessionsCompleted: 0
    },
    assessment: createInitialAssessmentState(),
    health: createInitialHealthState(),
    outcome: { ended: false },
    outreach: { consent: 'unknown', nudgeCount: 0 }
  };
}

function makeFakeGateway(): { gateway: LLMGateway; inputs: LLMInput[] } {
  const inputs: LLMInput[] = [];
  let index = 0;

  function nextOutput(): LLMOutput {
    index += 1;
    return {
      text: `reply_${index}`,
      provider: 'openai',
      model: 'gpt-test',
      latencyMs: 5,
      fallback: false,
      providerResponseId: `openai_resp_${index}`
    };
  }

  return {
    inputs,
    gateway: {
      async generateReply(input) {
        inputs.push(input);
        return nextOutput();
      },
      async streamReply(input, onToken) {
        inputs.push(input);
        const output = nextOutput();
        onToken(output.text, output.text);
        return output;
      }
    }
  };
}

function makeAbortAwareGateway(): LLMGateway {
  return {
    async generateReply() {
      return {
        text: 'unused',
        provider: 'heuristic',
        model: 'heuristic-test',
        latencyMs: 0,
        fallback: true
      };
    },
    async streamReply(_input, _onToken, options) {
      if (options?.signal?.aborted) {
        throw (options.signal.reason ?? new Error('request_aborted'));
      }
      throw new Error('stream_should_have_been_aborted');
    }
  };
}

function createService(persistence: InMemoryPersistence, gateway: LLMGateway) {
  return createGameService(gateway, persistence, {
    eventLogMax: 200,
    historyTurns: 8
  });
}

test('getSession projects wall time and applies compounded neglect', async () => {
  const sessionId = 'stale_projection';
  const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();
  const persistence = new InMemoryPersistence();
  const llm = makeFakeGateway();
  const service = createService(persistence, llm.gateway);

  persistence.records.set(sessionId, { state: makeState(sessionId, threeDaysAgo), events: [] });

  const record = await service.getSession(sessionId);
  assert.ok(record);
  assert.equal(record.state.timeline.currentDay, 4);
  assert.equal(record.state.health.value, 68);
  assert.equal(record.state.health.consecutiveMissedDays, 3);
  assert.equal(record.state.health.lastCause, 'neglect_penalty');
  assert.equal(persistence.upsertCalls, 1);
});

test('getSession can end run when wall-time neglect reaches death', async () => {
  const sessionId = 'walltime_death';
  const nineDaysAgo = new Date(Date.now() - 9 * 24 * 60 * 60 * 1000).toISOString();
  const persistence = new InMemoryPersistence();
  const llm = makeFakeGateway();
  const service = createService(persistence, llm.gateway);

  persistence.records.set(sessionId, { state: makeState(sessionId, nineDaysAgo), events: [] });

  const record = await service.getSession(sessionId);
  assert.ok(record);
  assert.equal(record.state.health.status, 'dead');
  assert.equal(record.state.health.value, 0);
  assert.equal(record.state.outcome.ended, true);
  assert.equal(record.state.outcome.reason, 'creature_died');
});

test('outreach nudge cooldown suppresses repeated notifications after mark sent', async () => {
  const sessionId = 'outreach_cooldown';
  const thirtyHoursAgo = new Date(Date.now() - 30 * 60 * 60 * 1000).toISOString();
  const persistence = new InMemoryPersistence();
  const llm = makeFakeGateway();
  const service = createService(persistence, llm.gateway);

  const state = makeState(sessionId, thirtyHoursAgo);
  state.timeline.lastInteractionAt = thirtyHoursAgo;
  persistence.records.set(sessionId, { state, events: [] });

  const firstNudge = await service.getOutreachNudge(sessionId);
  assert.ok(firstNudge);
  assert.equal(firstNudge.shouldNotify, true);
  assert.equal(firstNudge.severity, 'urgent');
  assert.equal(firstNudge.askForContact, true);

  const marked = await service.markOutreachSent(sessionId);
  assert.ok(marked);
  assert.equal(marked.outreach.nudgeCount, 1);
  assert.ok(marked.outreach.lastNudgeAt);

  const cooldownNudge = await service.getOutreachNudge(sessionId);
  assert.ok(cooldownNudge);
  assert.equal(cooldownNudge.shouldNotify, false);
  assert.equal(cooldownNudge.severity, 'none');
  assert.match(cooldownNudge.message, /Recent reminder already sent/i);
});

test('respondToInteraction persists events and forwards previous OpenAI response id', async () => {
  const sessionId = 'conversation_continuity';
  const nowIso = new Date().toISOString();
  const persistence = new InMemoryPersistence();
  const llm = makeFakeGateway();
  const service = createService(persistence, llm.gateway);

  persistence.records.set(sessionId, { state: makeState(sessionId, nowIso), events: [] });

  const first = await service.respondToInteraction(sessionId, 'thanks for helping me');
  assert.ok(first);
  assert.equal(first.state.progress.interactions, 1);
  assert.equal(first.state.timeline.interactionsToday, 1);
  assert.equal(first.event.providerResponseId, 'openai_resp_1');

  const second = await service.respondToInteraction(sessionId, 'can we keep going');
  assert.ok(second);
  assert.equal(second.state.progress.interactions, 2);
  assert.equal(persistence.appendEventCalls, 2);
  assert.equal(persistence.records.get(sessionId)?.events.length, 2);

  assert.equal(llm.inputs.length, 2);
  assert.equal(llm.inputs[1].previousProviderResponseId, 'openai_resp_1');
});

test('respondToInteraction rejects when session window is exhausted', async () => {
  const sessionId = 'window_exhausted';
  const nowIso = new Date().toISOString();
  const persistence = new InMemoryPersistence();
  const llm = makeFakeGateway();
  const service = createService(persistence, llm.gateway);

  const state = makeState(sessionId, nowIso);
  state.timeline.interactionsToday = state.timeline.interactionsPerSessionMax;
  persistence.records.set(sessionId, { state, events: [] });

  await assert.rejects(
    async () => {
      await service.respondToInteraction(sessionId, 'hello');
    },
    /session_window_exhausted/
  );
});

test('streamInteraction emits tokens and persists state/event on success', async () => {
  const sessionId = 'stream_success';
  const nowIso = new Date().toISOString();
  const persistence = new InMemoryPersistence();
  const llmInputs: LLMInput[] = [];
  const llm: LLMGateway = {
    async generateReply() {
      return {
        text: 'unused',
        provider: 'heuristic',
        model: 'heuristic-test',
        latencyMs: 0,
        fallback: true
      };
    },
    async streamReply(input, onToken) {
      llmInputs.push(input);
      onToken('hello ', 'hello ');
      onToken('there', 'hello there');
      return {
        text: 'hello there',
        provider: 'openai',
        model: 'gpt-test',
        latencyMs: 7,
        fallback: false,
        providerResponseId: 'openai_stream_resp_1'
      };
    }
  };
  const service = createService(persistence, llm);

  persistence.records.set(sessionId, { state: makeState(sessionId, nowIso), events: [] });

  const deltas: Array<{ delta: string; text: string }> = [];
  const result = await service.streamInteraction(
    sessionId,
    'thanks for being here',
    (delta, text) => deltas.push({ delta, text })
  );

  assert.ok(result);
  assert.equal(result.state.progress.interactions, 1);
  assert.equal(result.event.providerResponseId, 'openai_stream_resp_1');
  assert.equal(deltas.length, 2);
  assert.deepEqual(deltas[0], { delta: 'hello ', text: 'hello ' });
  assert.deepEqual(deltas[1], { delta: 'there', text: 'hello there' });
  assert.equal(persistence.appendEventCalls, 1);
  assert.equal(persistence.records.get(sessionId)?.events.length, 1);
  assert.equal(llmInputs.length, 1);
});

test('streamInteraction abort does not append event or mutate progress', async () => {
  const sessionId = 'stream_abort';
  const nowIso = new Date().toISOString();
  const persistence = new InMemoryPersistence();
  const service = createService(persistence, makeAbortAwareGateway());

  persistence.records.set(sessionId, { state: makeState(sessionId, nowIso), events: [] });

  const controller = new AbortController();
  controller.abort(new Error('stream_stopped_by_client'));

  await assert.rejects(
    async () => {
      await service.streamInteraction(sessionId, 'hello', () => {
        throw new Error('token_callback_should_not_fire');
      }, { signal: controller.signal });
    },
    /stream_stopped_by_client/
  );

  const stored = persistence.records.get(sessionId);
  assert.ok(stored);
  assert.equal(stored.events.length, 0);
  assert.equal(stored.state.progress.interactions, 0);
  assert.equal(stored.state.timeline.interactionsToday, 0);
  assert.equal(persistence.appendEventCalls, 0);
});
