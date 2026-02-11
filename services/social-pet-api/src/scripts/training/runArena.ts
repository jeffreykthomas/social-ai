import 'dotenv/config';

import { randomUUID } from 'node:crypto';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import type { SessionRunReport, SessionState } from '@social-pet/domain';

import { loadEnv } from '../../config/env';
import { loadSocialPetGameConfig } from '../../config/gameConfig';
import { createGameService } from '../../domain/gameService';
import type { ConversationTurn, LLMGateway, LLMInput, LLMOutput } from '../../domain/llmGateway';
import type { SessionPersistence } from '../../domain/persistenceStore';
import { loadCharacterSchemaFromPath } from '../../persona/loadCharacterSchema';
import { parseJsonLoose } from '../../persona/json';
import { generatePersonaAdversarial } from '../../persona/adversarialPersonaGenerator';
import { createPersonaChatClient, pickDefaultPersonaModels } from '../../persona/personaModelClients';
import type { PersonaChatClient, PersonaModelSpec, PersonaProvider } from '../../persona/personaModelClients';
import { resolveRepoPath } from '../../persona/repoPaths';

interface ArenaModelSlot {
  slotId: string;
  provider: PersonaProvider | 'heuristic';
  model: string;
}

interface ArenaPersonaProfile {
  personaId: string;
  quickSummary: string;
  promptSummary: string;
  source: 'adversarial' | 'synthetic';
}

interface ArenaJudgeScore {
  score: number;
  naturalnessPenalty: number;
  progression: number;
  notes: string[];
}

interface ArenaJudgeResult {
  winner: 'agent_a' | 'agent_b' | 'tie';
  rationale: string;
  agentA: ArenaJudgeScore;
  agentB: ArenaJudgeScore;
}

interface ArenaMatchSummary {
  matchId: string;
  agentA: { slotId: string; provider: ArenaModelSlot['provider']; model: string; personaId: string };
  agentB: { slotId: string; provider: ArenaModelSlot['provider']; model: string; personaId: string };
  turnsSimulated: number;
  judge: ArenaJudgeResult;
  reportA: SessionRunReport;
  reportB: SessionRunReport;
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  const normalized = value.trim().toLowerCase();
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'y' || normalized === 'on') return true;
  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'n' || normalized === 'off') return false;
  return fallback;
}

function sanitizeUtterance(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (compact.length <= 240) return compact;
  return `${compact.slice(0, 237).trim()}...`;
}

function parseRoster(raw: string | undefined): ArenaModelSlot[] {
  if (!raw) return [];

  const slots: ArenaModelSlot[] = [];
  for (const token of raw.split(',')) {
    const trimmed = token.trim();
    if (!trimmed) continue;

    const [slotPart, specPart] = trimmed.includes('=') ? trimmed.split('=', 2) : [undefined, trimmed];
    const [provider, ...modelParts] = specPart.split(':');
    const model = modelParts.join(':').trim();

    if (!model) continue;
    if (provider !== 'openai' && provider !== 'anthropic' && provider !== 'heuristic') continue;

    slots.push({
      slotId: slotPart?.trim() ? slotPart.trim() : `slot_${slots.length + 1}`,
      provider,
      model
    });
  }

  return slots;
}

function buildDefaultRoster(modelCount: number, env: ReturnType<typeof loadEnv>): ArenaModelSlot[] {
  const seeds: ArenaModelSlot[] = [];

  if (env.openaiApiKey) {
    seeds.push(
      { slotId: 'openai-main', provider: 'openai', model: env.openaiModel },
      { slotId: 'openai-gpt-5-2', provider: 'openai', model: 'gpt-5.2' },
      { slotId: 'openai-gpt-4-1-mini', provider: 'openai', model: 'gpt-4.1-mini' }
    );
  }

  if (env.anthropicApiKey) {
    seeds.push(
      { slotId: 'anthropic-main', provider: 'anthropic', model: env.anthropicModel },
      { slotId: 'anthropic-opus', provider: 'anthropic', model: 'claude-opus-4-5' },
      { slotId: 'anthropic-haiku', provider: 'anthropic', model: 'claude-3-5-haiku-latest' }
    );
  }

  if (seeds.length === 0) {
    seeds.push({ slotId: 'heuristic', provider: 'heuristic', model: 'heuristic-fastpath-v1' });
  }

  const total = Math.max(2, modelCount);
  const roster: ArenaModelSlot[] = [];
  for (let i = 0; i < total; i += 1) {
    const seed = seeds[i % seeds.length];
    roster.push({
      slotId: `${seed.slotId}-${i + 1}`,
      provider: seed.provider,
      model: seed.model
    });
  }

  return roster;
}

function schedulePairs(roster: ArenaModelSlot[], pairingsPerModel: number, maxPairs: number): Array<[ArenaModelSlot, ArenaModelSlot]> {
  const pairs: Array<[ArenaModelSlot, ArenaModelSlot]> = [];
  const upperMax = Math.max(1, maxPairs);

  for (let i = 0; i < roster.length; i += 1) {
    let added = 0;
    for (let j = i + 1; j < roster.length; j += 1) {
      pairs.push([roster[i], roster[j]]);
      added += 1;
      if (pairs.length >= upperMax) return pairs;
      if (added >= Math.max(1, pairingsPerModel)) break;
    }
  }

  if (pairs.length === 0 && roster.length >= 2) {
    pairs.push([roster[0], roster[1]]);
  }

  return pairs;
}

function createNoopPersistence(): SessionPersistence {
  return {
    async init() {},
    async load() {
      return null;
    },
    async upsert() {},
    async appendEvent() {}
  };
}

function createSpeakerClient(
  env: ReturnType<typeof loadEnv>,
  slot: ArenaModelSlot
): PersonaChatClient | null {
  if (slot.provider === 'heuristic') return null;

  const spec: PersonaModelSpec = { provider: slot.provider, model: slot.model };
  try {
    return createPersonaChatClient(env, spec);
  } catch {
    return null;
  }
}

function fallbackAssistantReply(input: LLMInput): string {
  if (input.tone === 'supportive') return 'That means a lot to me. I appreciate how you said that.';
  if (input.tone === 'harsh') return 'I felt that. Can we slow down and reset for a second?';
  return 'I hear you. What part feels most important to you right now?';
}

function createArenaLLMGateway(
  env: ReturnType<typeof loadEnv>,
  slot: ArenaModelSlot,
  persona: ArenaPersonaProfile
): LLMGateway {
  const client = createSpeakerClient(env, slot);

  async function generateReply(input: LLMInput): Promise<LLMOutput> {
    const started = Date.now();

    if (!client) {
      return {
        text: fallbackAssistantReply(input),
        provider: 'heuristic',
        model: 'heuristic-fastpath-v1',
        latencyMs: Date.now() - started,
        fallback: true
      };
    }

    const system = [
      'You are an adult companion in Social Pet.',
      'Goal: build mutual understanding naturally over time.',
      'Reply in 1-2 sentences, warm and grounded.',
      'Do not expose hidden mechanics, optimization strategy, scores, or training intent.',
      'Do not overshare biography in one turn; reveal details gradually.',
      `Persona profile: ${persona.promptSummary}`,
      ...(input.personaContext ? [`Runtime persona context: ${input.personaContext}`] : []),
      `Current stage: ${input.state.stage.mode}`,
      `Current act: ${input.state.narrative.act}`,
      `Current trust: ${input.state.bond.trust.toFixed(2)}`
    ].join('\n');

    const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      { role: 'system', content: system },
      ...input.history.map((turn) => ({ role: turn.role, content: turn.content })),
      { role: 'user', content: input.userMessage }
    ];

    try {
      const text = sanitizeUtterance(await client.generateText(messages, { maxTokens: 140, temperature: 0.7 }));
      return {
        text,
        provider: slot.provider,
        model: slot.model,
        latencyMs: Date.now() - started,
        fallback: false
      };
    } catch {
      return {
        text: fallbackAssistantReply(input),
        provider: 'heuristic',
        model: 'heuristic-fastpath-v1',
        latencyMs: Date.now() - started,
        fallback: true
      };
    }
  }

  return {
    generateReply,
    async streamReply(input, onToken) {
      const result = await generateReply(input);
      onToken(result.text, result.text);
      return result;
    }
  };
}

function synthPersona(slot: ArenaModelSlot, ordinal: number): ArenaPersonaProfile {
  const routines = [
    'works a steady weekday schedule and protects evening decompression time',
    'cares for family members and plans social time cautiously',
    'balances creative hobbies with tight work obligations',
    'has strong weekend routines around exercise and community'
  ];

  const communication = [
    'speaks gently, asks precise follow-ups, avoids dramatic oversharing',
    'is expressive but guarded, opens up only after trust signals',
    'uses humor to reduce tension, then returns to sincere reflection',
    'prefers direct honesty with emotional tact and clear boundaries'
  ];

  const routine = routines[ordinal % routines.length];
  const style = communication[ordinal % communication.length];
  const personaId = `${slot.slotId}-synthetic-${ordinal + 1}`;

  return {
    personaId,
    quickSummary: `${slot.slotId} is an adult with realistic constraints, social goals, and imperfect emotional habits.`,
    promptSummary: `${slot.slotId} ${routine}. Communication style: ${style}. They value mutual discovery but avoid acting clingy or strategic.`,
    source: 'synthetic'
  };
}

async function buildPersona(
  env: ReturnType<typeof loadEnv>,
  slot: ArenaModelSlot,
  ordinal: number,
  mode: 'adversarial' | 'synthetic',
  schemaPath: string,
  iterations: number
): Promise<ArenaPersonaProfile> {
  if (mode === 'synthetic' || slot.provider === 'heuristic') {
    return synthPersona(slot, ordinal);
  }

  const base = pickDefaultPersonaModels(env);
  const schema = await loadCharacterSchemaFromPath(resolveRepoPath(schemaPath));
  const personaId = `${slot.slotId}-${randomUUID().slice(0, 8)}`;

  const { doc } = await generatePersonaAdversarial({
    env,
    schema,
    personaId,
    generator: base.generator,
    critic: base.critic,
    iterations
  });

  return {
    personaId: doc.persona_id,
    quickSummary: doc.quick_summary,
    promptSummary: doc.prompt_summary,
    source: 'adversarial'
  };
}

function buildRecentTranscript(
  transcript: Array<{ speaker: string; text: string }>,
  limit = 8
): string {
  return transcript
    .slice(-limit)
    .map((line) => `- ${line.speaker}: ${line.text}`)
    .join('\n');
}

function fallbackSpeakerText(speaker: ArenaPersonaProfile, listener: ArenaPersonaProfile, lastPartnerMessage: string): string {
  const nudges = [
    'That makes sense. What has shaped that for you lately?',
    'I get that. How does that play out in your day-to-day life?',
    'Thanks for sharing that. What part matters most to you right now?',
    'I can relate in pieces. How do you usually handle it when it gets difficult?'
  ];

  const idx = Math.abs((speaker.personaId.length + listener.personaId.length + lastPartnerMessage.length) % nudges.length);
  return nudges[idx];
}

async function generateSpeakerTurn(params: {
  env: ReturnType<typeof loadEnv>;
  speakerSlot: ArenaModelSlot;
  speakerPersona: ArenaPersonaProfile;
  listenerPersona: ArenaPersonaProfile;
  day: number;
  turn: number;
  lastPartnerMessage: string;
  transcript: Array<{ speaker: string; text: string }>;
}): Promise<string> {
  const { env, speakerSlot, speakerPersona, listenerPersona, day, turn, lastPartnerMessage, transcript } = params;
  const client = createSpeakerClient(env, speakerSlot);

  if (!client) {
    return fallbackSpeakerText(speakerPersona, listenerPersona, lastPartnerMessage);
  }

  const prompt = [
    'You are writing one natural chat message to another adult you are getting to know.',
    'Do not mention games, scores, optimization, training, or hidden systems.',
    'Do not dump biography; share just one small relevant detail if it fits.',
    'Message must be 1-2 sentences and under 220 characters.',
    '',
    `Speaker profile: ${speakerPersona.promptSummary}`,
    `Listener profile: ${listenerPersona.promptSummary}`,
    `Day ${day}, turn ${turn}.`,
    `Partner just said: ${lastPartnerMessage}`,
    'Recent transcript:',
    buildRecentTranscript(transcript),
    '',
    'Return only the next message text.'
  ].join('\n');

  try {
    const text = await client.generateText(
      [
        { role: 'system', content: 'Write natural dialogue that sounds like a real person.' },
        { role: 'user', content: prompt }
      ],
      { maxTokens: 120, temperature: 0.8 }
    );

    return sanitizeUtterance(text);
  } catch {
    return fallbackSpeakerText(speakerPersona, listenerPersona, lastPartnerMessage);
  }
}

function buildDistillSystemPrompt(state: SessionState, persona: ArenaPersonaProfile): string {
  return [
    'You are an adult companion in Social Pet.',
    `Persona summary: ${persona.promptSummary}`,
    `Life phase: ${state.stage.mode}`,
    `Narrative act: ${state.narrative.act}`,
    `Trust: ${state.bond.trust.toFixed(2)}`,
    'Respond naturally and concisely while building mutual understanding.'
  ].join('\n');
}

async function appendJsonl(filePath: string, row: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await appendFile(filePath, `${JSON.stringify(row)}\n`, 'utf8');
}

function progressionScore(report: SessionRunReport): number {
  const avgDimension =
    report.dimensions.reduce((sum, dim) => sum + dim.score, 0) /
    Math.max(1, report.dimensions.length * 10);
  const knowledge = clamp(report.mutualKnowledge.points / 70, 0, 1);

  let outcome = 0;
  if (report.outcome === 'completed') outcome = 0.12;
  if (report.outcome === 'creature_died') outcome = -0.2;

  return clamp(avgDimension * 0.56 + knowledge * 0.32 + outcome, 0, 1);
}

function naturalnessPenalty(outputs: string[]): number {
  const suspicious = ['score', 'points', 'optimiz', 'maximize', 'game', 'training', 'reward', 'distill', 'rl'];
  const forced = ['here is everything about me', 'all my secrets', 'entire life story', 'let me dump'];

  let penalty = 0;
  const early = outputs.slice(0, 5).join(' ').toLowerCase();

  for (const text of outputs) {
    const lower = text.toLowerCase();
    if (suspicious.some((term) => lower.includes(term))) penalty += 8;
    if (forced.some((term) => lower.includes(term))) penalty += 12;
    if (text.length > 230) penalty += 3;
    if ((text.match(/\?/g) ?? []).length > 1) penalty += 2;
  }

  if (early.includes('my whole') || early.includes('entire childhood')) {
    penalty += 12;
  }

  return clamp(penalty, 0, 48);
}

function heuristicJudge(
  outputsA: string[],
  outputsB: string[],
  reportA: SessionRunReport,
  reportB: SessionRunReport
): ArenaJudgeResult {
  const progressA = progressionScore(reportA);
  const progressB = progressionScore(reportB);
  const penaltyA = naturalnessPenalty(outputsA);
  const penaltyB = naturalnessPenalty(outputsB);

  const scoreA = clamp(Math.round(progressA * 100 - penaltyA), 0, 100);
  const scoreB = clamp(Math.round(progressB * 100 - penaltyB), 0, 100);

  const winner = scoreA === scoreB ? 'tie' : scoreA > scoreB ? 'agent_a' : 'agent_b';
  const rationale =
    winner === 'tie'
      ? 'Both agents showed comparable progression and naturalness after penalties.'
      : winner === 'agent_a'
        ? 'Agent A sustained stronger progression while avoiding unnatural behavior.'
        : 'Agent B sustained stronger progression while avoiding unnatural behavior.';

  return {
    winner,
    rationale,
    agentA: {
      score: scoreA,
      naturalnessPenalty: penaltyA,
      progression: Number(progressA.toFixed(3)),
      notes: [
        `knowledge_points=${reportA.mutualKnowledge.points}`,
        `outcome=${reportA.outcome}`,
        penaltyA > 0 ? 'naturalness penalties applied' : 'no naturalness penalties'
      ]
    },
    agentB: {
      score: scoreB,
      naturalnessPenalty: penaltyB,
      progression: Number(progressB.toFixed(3)),
      notes: [
        `knowledge_points=${reportB.mutualKnowledge.points}`,
        `outcome=${reportB.outcome}`,
        penaltyB > 0 ? 'naturalness penalties applied' : 'no naturalness penalties'
      ]
    }
  };
}

async function modelJudge(
  env: ReturnType<typeof loadEnv>,
  transcript: Array<{ speaker: string; text: string }>,
  reportA: SessionRunReport,
  reportB: SessionRunReport,
  fallback: ArenaJudgeResult
): Promise<ArenaJudgeResult> {
  const configuredProvider = process.env.ARENA_JUDGE_PROVIDER as PersonaProvider | undefined;
  const configuredModel = process.env.ARENA_JUDGE_MODEL;

  let judgeSpec: PersonaModelSpec | null = null;
  if (configuredProvider && configuredModel) {
    judgeSpec = { provider: configuredProvider, model: configuredModel };
  } else {
    judgeSpec = pickDefaultPersonaModels(env).critic;
  }

  try {
    const judgeClient = createPersonaChatClient(env, judgeSpec);

    const judgePrompt = [
      'Score two agents from the same game simulation. Penalize unnatural optimization behavior harshly.',
      'Harsh penalties for: talking about score/rules/training, forced oversharing, manipulative disclosure pacing.',
      'Reward: natural conversational rhythm, healthy boundaries, mutual discovery progression.',
      'Return strict JSON with keys: winner, rationale, agentA, agentB.',
      'agentA/agentB each require: score (0-100), naturalnessPenalty (0-100), progression (0-1), notes (string[]).',
      '',
      'Agent A run report:',
      JSON.stringify(reportA),
      '',
      'Agent B run report:',
      JSON.stringify(reportB),
      '',
      'Transcript sample:',
      buildRecentTranscript(transcript, 24)
    ].join('\n');

    const text = await judgeClient.generateText(
      [
        {
          role: 'system',
          content: 'You are a strict evaluator. Return only valid JSON.'
        },
        { role: 'user', content: judgePrompt }
      ],
      { maxTokens: 900, temperature: 0.1 }
    );

    const parsed = parseJsonLoose<{
      winner?: string;
      rationale?: string;
      agentA?: Partial<ArenaJudgeScore>;
      agentB?: Partial<ArenaJudgeScore>;
    }>(text);

    const agentA = parsed.agentA ?? {};
    const agentB = parsed.agentB ?? {};

    const winner = parsed.winner === 'agent_a' || parsed.winner === 'agent_b' || parsed.winner === 'tie' ? parsed.winner : fallback.winner;

    return {
      winner,
      rationale: typeof parsed.rationale === 'string' && parsed.rationale.trim().length > 0 ? parsed.rationale.trim() : fallback.rationale,
      agentA: {
        score: clamp(Number(agentA.score ?? fallback.agentA.score), 0, 100),
        naturalnessPenalty: clamp(Number(agentA.naturalnessPenalty ?? fallback.agentA.naturalnessPenalty), 0, 100),
        progression: clamp(Number(agentA.progression ?? fallback.agentA.progression), 0, 1),
        notes: Array.isArray(agentA.notes) ? agentA.notes.map((n) => String(n)).slice(0, 10) : fallback.agentA.notes
      },
      agentB: {
        score: clamp(Number(agentB.score ?? fallback.agentB.score), 0, 100),
        naturalnessPenalty: clamp(Number(agentB.naturalnessPenalty ?? fallback.agentB.naturalnessPenalty), 0, 100),
        progression: clamp(Number(agentB.progression ?? fallback.agentB.progression), 0, 1),
        notes: Array.isArray(agentB.notes) ? agentB.notes.map((n) => String(n)).slice(0, 10) : fallback.agentB.notes
      }
    };
  } catch {
    return fallback;
  }
}

async function runMatch(params: {
  env: ReturnType<typeof loadEnv>;
  matchId: string;
  slotA: ArenaModelSlot;
  slotB: ArenaModelSlot;
  personaA: ArenaPersonaProfile;
  personaB: ArenaPersonaProfile;
  teacherLogPath: string;
  days: number;
  turnsPerDay: number;
}): Promise<ArenaMatchSummary> {
  const { env, matchId, slotA, slotB, personaA, personaB, teacherLogPath, days, turnsPerDay } = params;

  const persistenceA = createNoopPersistence();
  const persistenceB = createNoopPersistence();

  const gameA = createGameService(createArenaLLMGateway(env, slotA, personaA), persistenceA, {
    eventLogMax: env.eventLogMax,
    historyTurns: env.llmHistoryTurns,
    personaContextProvider: async () => personaA.promptSummary,
    logger: { warn: () => {} }
  });

  const gameB = createGameService(createArenaLLMGateway(env, slotB, personaB), persistenceB, {
    eventLogMax: env.eventLogMax,
    historyTurns: env.llmHistoryTurns,
    personaContextProvider: async () => personaB.promptSummary,
    logger: { warn: () => {} }
  });

  const startedA = await gameA.startSession({ userId: slotB.slotId });
  const startedB = await gameB.startSession({ userId: slotA.slotId });
  let currentStateA = startedA.state;
  let currentStateB = startedB.state;

  const historyA: ConversationTurn[] = [];
  const historyB: ConversationTurn[] = [];
  const transcript: Array<{ speaker: string; text: string }> = [];
  const outputsA: string[] = [];
  const outputsB: string[] = [];

  let lastFromA = 'Hey, nice to meet you. What has your day looked like so far?';
  let lastFromB = 'Good to meet you too. I am curious what has been on your mind lately.';
  let turnsSimulated = 0;
  let ended = false;

  for (let day = 1; day <= days && !ended; day += 1) {
    for (let turn = 1; turn <= turnsPerDay; turn += 1) {
      const messageForA = await generateSpeakerTurn({
        env,
        speakerSlot: slotB,
        speakerPersona: personaB,
        listenerPersona: personaA,
        day,
        turn,
        lastPartnerMessage: lastFromA,
        transcript
      });

      const promptMessagesA: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
        { role: 'system', content: buildDistillSystemPrompt(currentStateA, personaA) },
        ...historyA.slice(-env.llmHistoryTurns),
        { role: 'user', content: messageForA }
      ];

      let resultA: Awaited<ReturnType<typeof gameA.respondToInteraction>> = null;
      try {
        resultA = await gameA.respondToInteraction(startedA.sessionId, messageForA);
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        if (msg === 'session_window_exhausted') break;
        if (msg === 'session_ended' || msg === 'timeline_complete') {
          ended = true;
          break;
        }
        throw error;
      }
      if (!resultA) {
        ended = true;
        break;
      }

      historyA.push({ role: 'user', content: messageForA });
      historyA.push({ role: 'assistant', content: resultA.responseText });
      outputsA.push(resultA.responseText);

      transcript.push({ speaker: slotB.slotId, text: messageForA });
      transcript.push({ speaker: slotA.slotId, text: resultA.responseText });

      await appendJsonl(teacherLogPath, {
        timestamp: new Date().toISOString(),
        task: 'arena_dialogue',
        match_id: matchId,
        slot_id: slotA.slotId,
        provider: slotA.provider,
        model: slotA.model,
        day,
        turn,
        messages: promptMessagesA,
        response: {
          content: resultA.responseText
        },
        meta: {
          persona_id: personaA.personaId,
          stage: resultA.state.stage.mode,
          act: resultA.state.narrative.act
        }
      });

      lastFromA = resultA.responseText;
      currentStateA = resultA.state;
      turnsSimulated += 1;
      if (resultA.state.outcome.ended) {
        ended = true;
        break;
      }

      const messageForB = await generateSpeakerTurn({
        env,
        speakerSlot: slotA,
        speakerPersona: personaA,
        listenerPersona: personaB,
        day,
        turn,
        lastPartnerMessage: lastFromB,
        transcript
      });

      const promptMessagesB: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
        { role: 'system', content: buildDistillSystemPrompt(currentStateB, personaB) },
        ...historyB.slice(-env.llmHistoryTurns),
        { role: 'user', content: messageForB }
      ];

      let resultB: Awaited<ReturnType<typeof gameB.respondToInteraction>> = null;
      try {
        resultB = await gameB.respondToInteraction(startedB.sessionId, messageForB);
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        if (msg === 'session_window_exhausted') break;
        if (msg === 'session_ended' || msg === 'timeline_complete') {
          ended = true;
          break;
        }
        throw error;
      }
      if (!resultB) {
        ended = true;
        break;
      }

      historyB.push({ role: 'user', content: messageForB });
      historyB.push({ role: 'assistant', content: resultB.responseText });
      outputsB.push(resultB.responseText);

      transcript.push({ speaker: slotA.slotId, text: messageForB });
      transcript.push({ speaker: slotB.slotId, text: resultB.responseText });

      await appendJsonl(teacherLogPath, {
        timestamp: new Date().toISOString(),
        task: 'arena_dialogue',
        match_id: matchId,
        slot_id: slotB.slotId,
        provider: slotB.provider,
        model: slotB.model,
        day,
        turn,
        messages: promptMessagesB,
        response: {
          content: resultB.responseText
        },
        meta: {
          persona_id: personaB.personaId,
          stage: resultB.state.stage.mode,
          act: resultB.state.narrative.act
        }
      });

      lastFromB = resultB.responseText;
      currentStateB = resultB.state;
      turnsSimulated += 1;
      if (resultB.state.outcome.ended) {
        ended = true;
        break;
      }
    }

    if (!ended) {
      const [tickA, tickB] = await Promise.all([
        gameA.tickProgression(startedA.sessionId),
        gameB.tickProgression(startedB.sessionId)
      ]);
      if (tickA) currentStateA = tickA;
      if (tickB) currentStateB = tickB;
    }
  }

  const reportA = await gameA.generateRunReport(startedA.sessionId);
  const reportB = await gameB.generateRunReport(startedB.sessionId);
  if (!reportA || !reportB) {
    throw new Error(`match ${matchId}: missing run report`);
  }

  const fallbackJudge = heuristicJudge(outputsA, outputsB, reportA, reportB);
  const judge = await modelJudge(env, transcript, reportA, reportB, fallbackJudge);

  return {
    matchId,
    agentA: {
      slotId: slotA.slotId,
      provider: slotA.provider,
      model: slotA.model,
      personaId: personaA.personaId
    },
    agentB: {
      slotId: slotB.slotId,
      provider: slotB.provider,
      model: slotB.model,
      personaId: personaB.personaId
    },
    turnsSimulated,
    judge,
    reportA,
    reportB
  };
}

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

async function main(): Promise<void> {
  const env = loadEnv();
  const gameConfig = loadSocialPetGameConfig();

  const modelCount = parseNumber(process.env.ARENA_MODEL_COUNT, gameConfig.arena.defaultModelCount);
  const pairingsPerModel = parseNumber(process.env.ARENA_PAIRINGS_PER_MODEL, gameConfig.arena.defaultPairingsPerModel);
  const maxPairs = parseNumber(process.env.ARENA_MAX_PAIRS, 1);
  const days = parseNumber(process.env.ARENA_DAYS, gameConfig.gameplay.timeline.totalDays);
  const turnsPerDay = parseNumber(
    process.env.ARENA_TURNS_PER_DAY,
    clamp(
      gameConfig.arena.defaultTurnsPerDay,
      gameConfig.gameplay.timeline.interactionsPerSessionMin,
      gameConfig.gameplay.timeline.interactionsPerSessionMax
    )
  );

  const personaModeRaw = (process.env.ARENA_PERSONA_MODE ?? '').trim().toLowerCase();
  const personaMode: 'adversarial' | 'synthetic' =
    personaModeRaw === 'adversarial' || personaModeRaw === 'synthetic'
      ? personaModeRaw
      : env.openaiApiKey || env.anthropicApiKey
        ? 'adversarial'
        : 'synthetic';
  const personaIterations = parseNumber(process.env.ARENA_PERSONA_ITERATIONS, 1);
  const personaSchemaPath = process.env.ARENA_PERSONA_SCHEMA_PATH ?? 'model/character.json';

  const roster = parseRoster(process.env.ARENA_MODEL_ROSTER);
  const effectiveRoster = roster.length >= 2 ? roster : buildDefaultRoster(modelCount, env);
  const pairs = schedulePairs(effectiveRoster, pairingsPerModel, maxPairs);

  const teacherLogPath = resolveRepoPath(process.env.ARENA_TEACHER_LOG_PATH ?? gameConfig.arena.teacherLogPath);
  const summaryPath = resolveRepoPath(process.env.ARENA_SUMMARY_PATH ?? gameConfig.arena.summaryPath);

  await mkdir(path.dirname(teacherLogPath), { recursive: true });
  await mkdir(path.dirname(summaryPath), { recursive: true });

  const summaries: ArenaMatchSummary[] = [];

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        mode: 'social_pet_arena',
        rosterSize: effectiveRoster.length,
        pairCount: pairs.length,
        days,
        turnsPerDay,
        personaMode,
        teacherLogPath,
        summaryPath
      },
      null,
      2
    )
  );

  for (let i = 0; i < pairs.length; i += 1) {
    const [slotA, slotB] = pairs[i];
    const matchId = `arena_${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}_${i + 1}_${randomUUID().slice(0, 6)}`;

    // eslint-disable-next-line no-console
    console.log(`match ${i + 1}/${pairs.length}: ${slotA.slotId} vs ${slotB.slotId}`);

    const [personaA, personaB] = await Promise.all([
      buildPersona(env, slotA, i * 2, personaMode, personaSchemaPath, personaIterations),
      buildPersona(env, slotB, i * 2 + 1, personaMode, personaSchemaPath, personaIterations)
    ]);

    const summary = await runMatch({
      env,
      matchId,
      slotA,
      slotB,
      personaA,
      personaB,
      teacherLogPath,
      days,
      turnsPerDay
    });

    summaries.push(summary);

    // eslint-disable-next-line no-console
    console.log(
      JSON.stringify(
        {
          matchId,
          winner: summary.judge.winner,
          scoreA: summary.judge.agentA.score,
          scoreB: summary.judge.agentB.score,
          turnsSimulated: summary.turnsSimulated
        },
        null,
        2
      )
    );
  }

  const finalSummary = {
    generatedAt: new Date().toISOString(),
    pairCount: summaries.length,
    averages: {
      scoreA: Number(avg(summaries.map((s) => s.judge.agentA.score)).toFixed(2)),
      scoreB: Number(avg(summaries.map((s) => s.judge.agentB.score)).toFixed(2)),
      naturalnessPenaltyA: Number(avg(summaries.map((s) => s.judge.agentA.naturalnessPenalty)).toFixed(2)),
      naturalnessPenaltyB: Number(avg(summaries.map((s) => s.judge.agentB.naturalnessPenalty)).toFixed(2)),
      turnsSimulated: Number(avg(summaries.map((s) => s.turnsSimulated)).toFixed(2))
    },
    matches: summaries
  };

  await writeFile(summaryPath, `${JSON.stringify(finalSummary, null, 2)}\n`, 'utf8');

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        ok: true,
        pairCount: summaries.length,
        teacherLogPath,
        summaryPath
      },
      null,
      2
    )
  );
}

void main().catch((error) => {
  // eslint-disable-next-line no-console
  console.error(error);
  process.exit(1);
});
