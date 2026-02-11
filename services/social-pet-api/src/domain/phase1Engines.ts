import type {
  AssessmentDimensionKey,
  AssessmentDimensionState,
  CharacterFacetKey,
  CharacterFacetReveal,
  CreatureHealthState,
  GameTimelineState,
  HealthStatus,
  HeroJourneyAct,
  InteractionEvent,
  InteractionTone,
  MutualKnowledgeReport,
  MutualKnowledgeState,
  OutreachConsent,
  OutreachNudge,
  PlayerAssessmentState,
  SessionOutcomeState,
  SessionRunReport,
  StageMode,
  UserFact,
  UserFactCategory
} from '@social-pet/domain';
import { loadSocialPetGameConfig } from '../config/gameConfig';

const assessmentKeys: AssessmentDimensionKey[] = [
  'warmth',
  'confrontation_comfort',
  'autonomy_support',
  'emotional_attunement'
];

const GAME_CONFIG = loadSocialPetGameConfig();
const TIMELINE_CONFIG = GAME_CONFIG.gameplay.timeline;
const STAGE_GATES = GAME_CONFIG.gameplay.stageGates;
const ACT_CONFIG = GAME_CONFIG.gameplay.acts;
const TRIAL_WINDOW = GAME_CONFIG.gameplay.trialWindow;
const MS_PER_DAY = 24 * 60 * 60 * 1000;

function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function clampScore(value: number): number {
  return clamp(value, 1, 10);
}

function includesAny(text: string, phrases: string[]): boolean {
  return phrases.some((phrase) => text.includes(phrase));
}

function confidenceForSamples(samples: number): number {
  if (samples <= 0) return 0.15;
  const normalized = Math.log1p(samples) / Math.log(12);
  return clamp(0.2 + normalized * 0.8, 0.15, 1);
}

export function createInitialTimelineState(nowIso = new Date().toISOString()): GameTimelineState {
  return {
    totalDays: TIMELINE_CONFIG.totalDays,
    currentDay: 1,
    sessionWindowMinutes: TIMELINE_CONFIG.sessionWindowMinutes,
    interactionsPerSessionMin: TIMELINE_CONFIG.interactionsPerSessionMin,
    interactionsPerSessionMax: TIMELINE_CONFIG.interactionsPerSessionMax,
    interactionsToday: 0,
    sessionsCompleted: 0,
    startedAt: nowIso,
    dayStartedAt: nowIso
  };
}

export function wholeDaysBetween(startIso: string, endIso: string): number {
  const startMs = Date.parse(startIso);
  const endMs = Date.parse(endIso);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return 0;
  if (endMs <= startMs) return 0;
  return Math.floor((endMs - startMs) / MS_PER_DAY);
}

export function hoursBetween(startIso: string, endIso: string): number {
  const startMs = Date.parse(startIso);
  const endMs = Date.parse(endIso);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return 0;
  if (endMs <= startMs) return 0;
  return (endMs - startMs) / (60 * 60 * 1000);
}

function makeDimensionState(key: AssessmentDimensionKey): AssessmentDimensionState {
  return {
    key,
    score: 5,
    confidence: 0.15,
    evidenceCount: 0,
    evidence: []
  };
}

export function createInitialAssessmentState(): PlayerAssessmentState {
  return {
    dimensions: {
      warmth: makeDimensionState('warmth'),
      confrontation_comfort: makeDimensionState('confrontation_comfort'),
      autonomy_support: makeDimensionState('autonomy_support'),
      emotional_attunement: makeDimensionState('emotional_attunement')
    },
    totalEvidence: 0,
    uncertainDimensions: [...assessmentKeys]
  };
}

export function createInitialKnowledgeState(): MutualKnowledgeState {
  return {
    points: 0,
    userFacts: [],
    characterFacetsRevealed: []
  };
}

export function createInitialHealthState(): CreatureHealthState {
  return {
    value: 80,
    status: 'healthy',
    lastDelta: 0,
    lastCause: 'session_start',
    consecutiveSupportive: 0,
    consecutiveHarsh: 0,
    consecutiveMissedDays: 0
  };
}

export function healthStatusForValue(value: number): HealthStatus {
  if (value <= 0) return 'dead';
  if (value <= 19) return 'dying';
  if (value <= 39) return 'withering';
  if (value <= 59) return 'wilting';
  return 'healthy';
}

export function canAcceptInteraction(timeline: GameTimelineState, outcome: SessionOutcomeState): { ok: boolean; reason?: string } {
  if (outcome.ended) return { ok: false, reason: 'session_ended' };
  if (timeline.currentDay > timeline.totalDays) return { ok: false, reason: 'timeline_complete' };
  if (timeline.interactionsToday >= TIMELINE_CONFIG.interactionsHardCap) {
    return { ok: false, reason: 'session_window_exhausted' };
  }
  return { ok: true };
}

export type ConversationUrgency = 'normal' | 'winding_down' | 'final' | 'overtime';

export function conversationUrgency(timeline: GameTimelineState): ConversationUrgency {
  if (timeline.interactionsToday > timeline.interactionsPerSessionMax) return 'overtime';
  if (timeline.interactionsToday === timeline.interactionsPerSessionMax) return 'final';
  if (timeline.interactionsToday === timeline.interactionsPerSessionMax - 1) return 'winding_down';
  return 'normal';
}

export function registerInteractionInTimeline(
  timeline: GameTimelineState,
  nowIso = new Date().toISOString()
): GameTimelineState {
  const isFirstInteractionToday = timeline.interactionsToday === 0;
  return {
    ...timeline,
    interactionsToday: timeline.interactionsToday + 1,
    sessionsCompleted: isFirstInteractionToday ? timeline.sessionsCompleted + 1 : timeline.sessionsCompleted,
    lastInteractionAt: nowIso
  };
}

export function advanceTimelineDay(
  timeline: GameTimelineState,
  nowIso = new Date().toISOString()
): { timeline: GameTimelineState; hadInteractionToday: boolean } {
  const hadInteractionToday = timeline.interactionsToday > 0;
  const nextDay = Math.min(timeline.totalDays, timeline.currentDay + 1);

  return {
    timeline: {
      ...timeline,
      currentDay: nextDay,
      interactionsToday: 0,
      dayStartedAt: nowIso
    },
    hadInteractionToday
  };
}

export function stageForTimeline(day: number, totalInteractions: number): StageMode {
  if (day >= STAGE_GATES.old.minDay && totalInteractions >= STAGE_GATES.old.minInteractions) return 'old';
  if (day >= STAGE_GATES.wise.minDay && totalInteractions >= STAGE_GATES.wise.minInteractions) return 'wise';
  if (day >= STAGE_GATES.middle_aged.minDay && totalInteractions >= STAGE_GATES.middle_aged.minInteractions) {
    return 'middle_aged';
  }
  return 'young_adult';
}

export function actForTimelineDay(day: number): HeroJourneyAct {
  if (day >= ACT_CONFIG.integrationStartDay) return 'integration';
  if (day >= ACT_CONFIG.trialsAndFrictionStartDay) return 'trials_and_friction';
  return 'safe_bonding';
}

export function activeTrialForTimeline(day: number, healthStatus: HealthStatus): string | null {
  if (day < TRIAL_WINDOW.startDay || day > TRIAL_WINDOW.endDay) return null;
  if (healthStatus === 'withering' || healthStatus === 'dying') return 'survival_repair';
  if (day <= TRIAL_WINDOW.earlyRepairCutoffDay) return 'identity_crisis';
  return 'repair_trials';
}

type AssessmentObservation = {
  scores: Record<AssessmentDimensionKey, number>;
  signals: string[];
};

function observeAssessment(
  message: string,
  tone: InteractionTone,
  targeted: AssessmentDimensionKey[]
): AssessmentObservation {
  const text = message.toLowerCase();
  const signals: string[] = [];

  let warmth = 5;
  if (tone === 'supportive') warmth += 2;
  if (tone === 'harsh') warmth -= 2.5;
  if (includesAny(text, ['thank', 'thanks', 'sorry', 'care', 'love', 'proud'])) {
    warmth += 1.5;
    signals.push('warm language detected');
  }
  if (includesAny(text, ['stupid', 'shut up', 'whatever', 'hate'])) {
    warmth -= 2;
    signals.push('dismissive language detected');
  }

  let confrontationComfort = 5;
  if (includesAny(text, ['let us talk', "let's talk", 'disagree', 'because', 'boundary', 'we need'])) {
    confrontationComfort += 2;
    signals.push('direct conflict engagement');
  }
  if (includesAny(text, ['ignore it', 'never mind', 'drop it', 'fine'])) {
    confrontationComfort -= 1.5;
    signals.push('conflict avoidance marker');
  }
  if (tone === 'harsh') confrontationComfort += 0.5;

  let autonomySupport = 5;
  if (includesAny(text, ['your choice', 'you choose', 'what do you want', 'decide together'])) {
    autonomySupport += 2.5;
    signals.push('agency-support language');
  }
  if (includesAny(text, ['because i said', 'you must', 'do what i say'])) {
    autonomySupport -= 3;
    signals.push('controlling language marker');
  }

  let emotionalAttunement = 5;
  if (includesAny(text, ['i hear you', 'you feel', 'that sounds hard', 'that must feel'])) {
    emotionalAttunement += 2.5;
    signals.push('emotion acknowledgment');
  }
  if (includesAny(text, ['calm down', 'get over it', 'stop being'])) {
    emotionalAttunement -= 2;
    signals.push('emotion dismissal');
  }

  for (const key of targeted) {
    if (key === 'warmth') warmth += 0.6;
    if (key === 'confrontation_comfort') confrontationComfort += 0.6;
    if (key === 'autonomy_support') autonomySupport += 0.6;
    if (key === 'emotional_attunement') emotionalAttunement += 0.6;
  }

  return {
    scores: {
      warmth: clampScore(warmth),
      confrontation_comfort: clampScore(confrontationComfort),
      autonomy_support: clampScore(autonomySupport),
      emotional_attunement: clampScore(emotionalAttunement)
    },
    signals
  };
}

export function applyAssessmentObservation(
  assessment: PlayerAssessmentState,
  message: string,
  tone: InteractionTone,
  eventId: string,
  atIso: string,
  targeted: AssessmentDimensionKey[],
  interactionId?: string
): { assessment: PlayerAssessmentState; scores: Partial<Record<AssessmentDimensionKey, number>>; signals: string[] } {
  const observed = observeAssessment(message, tone, targeted);
  const nextDimensions = { ...assessment.dimensions };

  for (const key of assessmentKeys) {
    const current = nextDimensions[key];
    const observedScore = observed.scores[key];
    const nextEvidenceCount = current.evidenceCount + 1;
    const nextScore = (current.score * current.evidenceCount + observedScore) / nextEvidenceCount;
    const nextConfidence = confidenceForSamples(nextEvidenceCount);
    const nextEvidence = [
      ...current.evidence,
      {
        eventId,
        at: atIso,
        score: observedScore,
        note: observed.signals[0] ?? `observed ${key}`,
        interactionId
      }
    ].slice(-24);

    nextDimensions[key] = {
      ...current,
      score: clampScore(nextScore),
      confidence: nextConfidence,
      evidenceCount: nextEvidenceCount,
      evidence: nextEvidence
    };
  }

  const uncertainDimensions = assessmentKeys.filter((key) => nextDimensions[key].confidence < 0.6);

  return {
    assessment: {
      dimensions: nextDimensions,
      totalEvidence: assessment.totalEvidence + 1,
      uncertainDimensions
    },
    scores: observed.scores,
    signals: observed.signals
  };
}

export function applyHealthFromInteraction(
  health: CreatureHealthState,
  tone: InteractionTone
): { health: CreatureHealthState; delta: number; cause: string } {
  let delta = 0;
  let cause = 'neutral_interaction';

  if (tone === 'supportive') {
    delta = 4 + Math.min(health.consecutiveSupportive, 2);
    if (health.status === 'withering' || health.status === 'dying') {
      delta += health.consecutiveSupportive >= 1 ? 4 : 2;
      cause = 'recovery_care';
    } else {
      cause = 'supportive_interaction';
    }
  } else if (tone === 'harsh') {
    delta = -8 - Math.min(health.consecutiveHarsh * 2, 6);
    cause = 'harmful_interaction';
  } else {
    delta = 1;
  }

  const nextValue = clamp(health.value + delta, 0, 100);
  const nextStatus = healthStatusForValue(nextValue);
  const nextSupportive = tone === 'supportive' ? health.consecutiveSupportive + 1 : 0;
  const nextHarsh = tone === 'harsh' ? health.consecutiveHarsh + 1 : 0;

  return {
    health: {
      ...health,
      value: nextValue,
      status: nextStatus,
      lastDelta: delta,
      lastCause: cause,
      consecutiveSupportive: nextSupportive,
      consecutiveHarsh: nextHarsh,
      consecutiveMissedDays: 0
    },
    delta,
    cause
  };
}

export function applyHealthDayAdvance(
  health: CreatureHealthState,
  hadInteractionToday: boolean
): { health: CreatureHealthState; delta: number; cause: string } {
  if (hadInteractionToday) {
    return {
      health: { ...health, consecutiveMissedDays: 0, lastDelta: 0, lastCause: 'day_advance' },
      delta: 0,
      cause: 'day_advance'
    };
  }

  const nextMissed = health.consecutiveMissedDays + 1;
  const compoundedPenalty = Math.min(5 + Math.max(0, nextMissed - 2) * 2, 17);
  const delta = nextMissed > 1 ? -compoundedPenalty : 0;
  const cause = nextMissed > 1 ? 'neglect_penalty' : 'grace_day';
  const nextValue = clamp(health.value + delta, 0, 100);

  return {
    health: {
      ...health,
      value: nextValue,
      status: healthStatusForValue(nextValue),
      consecutiveMissedDays: nextMissed,
      lastDelta: delta,
      lastCause: cause,
      consecutiveSupportive: 0,
      consecutiveHarsh: 0
    },
    delta,
    cause
  };
}

export function deriveOutcome(
  previous: SessionOutcomeState,
  nowIso: string,
  stage: StageMode,
  timeline: GameTimelineState,
  healthStatus: HealthStatus,
  totalInteractions: number
): SessionOutcomeState {
  if (previous.ended) return previous;

  if (healthStatus === 'dead') {
    return { ended: true, reason: 'creature_died', endedAt: nowIso };
  }

  if (timeline.currentDay >= timeline.totalDays && stage === 'old' && totalInteractions >= 45) {
    return { ended: true, reason: 'completed', endedAt: nowIso };
  }

  return previous;
}

function normalizeFactKey(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]+/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function takeFirstSentence(text: string, maxLen = 160): string {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  const sentence = cleaned.split(/(?<=[.!?])\s+/)[0] ?? cleaned;
  return sentence.length <= maxLen ? sentence : `${sentence.slice(0, maxLen - 3).trim()}...`;
}

function extractUserFacts(message: string): Array<{ category: UserFactCategory; fact: string }> {
  const lower = message.trim().toLowerCase();
  const facts: Array<{ category: UserFactCategory; fact: string }> = [];

  const ageMatch = lower.match(/\b(i am|i'm)\s+(\d{1,3})\b/);
  if (ageMatch) facts.push({ category: 'demographics', fact: `age ${ageMatch[2]}` });

  const liveMatch = lower.match(/\b(i live in|i'm in|i am in|based in)\s+([^.,;!?\n]+)/);
  if (liveMatch) facts.push({ category: 'location', fact: `lives in ${liveMatch[2].trim()}` });

  const fromMatch = lower.match(/\b(i am from|i'm from)\s+([^.,;!?\n]+)/);
  if (fromMatch) facts.push({ category: 'location', fact: `from ${fromMatch[2].trim()}` });

  const workMatch = lower.match(/\b(i work as|i work at|i do)\s+([^.,;!?\n]+)/);
  if (workMatch) facts.push({ category: 'work', fact: `work: ${workMatch[2].trim()}` });

  const jobMatch = lower.match(/\bmy job is\s+([^.,;!?\n]+)/);
  if (jobMatch) facts.push({ category: 'work', fact: `job: ${jobMatch[1].trim()}` });

  const likes = lower.match(/\b(i like|i love|i enjoy)\s+([^.,;!?\n]+)/);
  if (likes) facts.push({ category: 'preference', fact: `likes ${likes[2].trim()}` });

  const fave = lower.match(/\bmy favorite\s+([^.,;!?\n]+)/);
  if (fave) facts.push({ category: 'preference', fact: `favorite ${fave[1].trim()}` });

  const goal = lower.match(/\b(i want to|i hope to|i'm trying to|i am trying to)\s+([^.,;!?\n]+)/);
  if (goal) facts.push({ category: 'goal', fact: `wants to ${goal[2].trim()}` });

  const haveMatch = lower.match(/\b(i have)\s+([^.,;!?\n]+)/);
  if (haveMatch && /\b(kid|kids|child|children|dog|cat|pets?|partner|wife|husband|fianc|fiancé)\b/.test(haveMatch[2])) {
    facts.push({ category: 'relationship', fact: `has ${haveMatch[2].trim()}` });
  }

  const identity = lower.match(/\b(i am|i'm)\s+(a|an)\s+([^.,;!?\n]+)/);
  if (identity) facts.push({ category: 'identity', fact: `is ${identity[2]} ${identity[3].trim()}` });

  const seen = new Set<string>();
  return facts.filter((f) => {
    const key = normalizeFactKey(`${f.category}:${f.fact}`);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

type FacetHit = { facet: CharacterFacetKey; evidence: string };

function inferCharacterFacetReveals(responseText: string): FacetHit[] {
  const text = responseText.trim();
  const lower = text.toLowerCase();
  const hits: FacetHit[] = [];

  const add = (facet: CharacterFacetKey) => {
    hits.push({ facet, evidence: takeFirstSentence(text) });
  };

  if (/\b(i value|i care about|it matters to me|i believe)\b/.test(lower)) add('values_beliefs_ethics');
  if (/\b(i want|i need|i crave|i'm looking for|i am looking for|i'm drawn to)\b/.test(lower)) add('motivations_needs_drives');
  if (/\b(i prefer|i tend to|in groups i|i'm more of a)\b/.test(lower)) add('social_style_relationship_patterns');
  if (/\b(when i'm stressed|when i get stressed|i cope|i calm down|i get anxious|i get irritable)\b/.test(lower)) {
    add('emotion_affect_regulation');
  }
  if (/\b(i'm direct|i try to be direct|i'm blunt|i keep it short|i ramble|i'm not great at words)\b/.test(lower)) {
    add('communication_voice_expression');
  }
  if (/\b(boundar(y|ies)|i need space|i get close|i trust slowly|i open up)\b/.test(lower)) add('attachment_intimacy_boundaries');
  if (/\b(in conflict|when we disagree|i argue|i negotiate|i don't like power games)\b/.test(lower)) add('conflict_power_strategy');
  if (/\b(i'm trying to learn|i'm working on|i used to|i have changed|i've changed|i want to be better)\b/.test(lower)) {
    add('growth_arc_change_dynamics');
  }

  const seen = new Set<CharacterFacetKey>();
  return hits.filter((h) => {
    if (seen.has(h.facet)) return false;
    seen.add(h.facet);
    return true;
  });
}

export function applyMutualKnowledgeFromInteraction(
  knowledge: MutualKnowledgeState,
  event: Pick<InteractionEvent, 'id' | 'at' | 'userMessage' | 'responseText'>
): { knowledge: MutualKnowledgeState; newPoints: number } {
  const nextFacts = [...knowledge.userFacts];
  const nextReveals = [...knowledge.characterFacetsRevealed];

  const existingUserFactKeys = new Set(nextFacts.map((f) => normalizeFactKey(`${f.category}:${f.fact}`)));
  const existingFacetKeys = new Set(nextReveals.map((r) => r.facet));

  let newPoints = 0;

  const extractedFacts = extractUserFacts(event.userMessage);
  extractedFacts.forEach((fact, idx) => {
    const key = normalizeFactKey(`${fact.category}:${fact.fact}`);
    if (existingUserFactKeys.has(key)) return;
    existingUserFactKeys.add(key);
    const entry: UserFact = {
      id: `uf_${event.id}_${idx}`,
      at: event.at,
      category: fact.category,
      fact: fact.fact,
      eventId: event.id
    };
    nextFacts.push(entry);
    newPoints += 1;
  });

  const facetHits = inferCharacterFacetReveals(event.responseText);
  facetHits.forEach((hit, idx) => {
    if (existingFacetKeys.has(hit.facet)) return;
    existingFacetKeys.add(hit.facet);
    const entry: CharacterFacetReveal = {
      id: `cf_${event.id}_${idx}`,
      at: event.at,
      facet: hit.facet,
      evidence: hit.evidence,
      eventId: event.id
    };
    nextReveals.push(entry);
    newPoints += 1;
  });

  const MAX_USER_FACTS = 300;
  const MAX_FACET_REVEALS = 64;

  const cappedFacts = nextFacts.length > MAX_USER_FACTS ? nextFacts.slice(nextFacts.length - MAX_USER_FACTS) : nextFacts;
  const cappedReveals =
    nextReveals.length > MAX_FACET_REVEALS ? nextReveals.slice(nextReveals.length - MAX_FACET_REVEALS) : nextReveals;

  return {
    knowledge: {
      points: knowledge.points + newPoints,
      userFacts: cappedFacts,
      characterFacetsRevealed: cappedReveals
    },
    newPoints
  };
}

export function buildMutualKnowledgeReport(knowledge: MutualKnowledgeState): MutualKnowledgeReport {
  const counts = new Map<UserFactCategory, number>();
  for (const fact of knowledge.userFacts) {
    counts.set(fact.category, (counts.get(fact.category) ?? 0) + 1);
  }

  const topUserFactCategories = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([category, count]) => ({ category, count }));

  const revealedFacets = Array.from(new Set(knowledge.characterFacetsRevealed.map((r) => r.facet)));
  const sampleUserFacts = knowledge.userFacts.slice(-12);

  return {
    points: knowledge.points,
    userFactsCount: knowledge.userFacts.length,
    characterFacetsRevealedCount: revealedFacets.length,
    topUserFactCategories,
    characterFacetsRevealed: revealedFacets,
    sampleUserFacts
  };
}

function interpretDimension(score: number, confidence: number): string {
  const confidenceLabel = confidence >= 0.75 ? 'high confidence' : confidence >= 0.55 ? 'medium confidence' : 'early signal';
  if (score >= 7.5) return `strong tendency (${confidenceLabel})`;
  if (score >= 5.5) return `balanced tendency (${confidenceLabel})`;
  return `lower tendency (${confidenceLabel})`;
}

export function buildRunReport(stateSessionId: string, state: {
  stageMode: StageMode;
  currentDay: number;
  healthStatus: HealthStatus;
  healthValue: number;
  outcome: SessionOutcomeState;
  assessment: PlayerAssessmentState;
  knowledge: MutualKnowledgeState;
}, events: InteractionEvent[]): SessionRunReport {
  const generatedAt = new Date().toISOString();
  const outcome = state.outcome.ended ? (state.outcome.reason ?? 'in_progress') : 'in_progress';
  const mutualKnowledge = buildMutualKnowledgeReport(state.knowledge);

  const dimensions = assessmentKeys.map((key) => {
    const dim = state.assessment.dimensions[key];
    return {
      key,
      score: Number(dim.score.toFixed(2)),
      confidence: Number(dim.confidence.toFixed(2)),
      interpretation: interpretDimension(dim.score, dim.confidence)
    };
  });

  const strongest = [...dimensions].sort((a, b) => b.score - a.score).slice(0, 2).map((d) => {
    return `${d.key.replace('_', ' ')} appeared as a relative strength (score ${d.score.toFixed(1)}).`;
  });

  const supportiveCount = events.filter((event) => event.tone === 'supportive').length;
  const harshCount = events.filter((event) => event.tone === 'harsh').length;

  const patterns: string[] = [
    `Supportive interactions: ${supportiveCount}; harsh interactions: ${harshCount}.`,
    `Current health state: ${state.healthStatus} (${state.healthValue}/100).`,
    `Progressed to ${state.stageMode} by day ${state.currentDay}.`,
    `Mutual knowledge points: ${mutualKnowledge.points} (learned ${mutualKnowledge.userFactsCount} user facts; character revealed ${mutualKnowledge.characterFacetsRevealedCount} facets).`
  ];

  if (harshCount > supportiveCount) {
    patterns.push('Conflict pressure outweighed repair in this run.');
  } else if (supportiveCount > 0) {
    patterns.push('Repair/support moments were a stabilizing force across sessions.');
  }

  const confidenceNotes = dimensions
    .filter((dimension) => dimension.confidence < 0.6)
    .map((dimension) => {
      return `${dimension.key.replace('_', ' ')} still has low confidence; gather more varied interactions.`;
    });

  if (confidenceNotes.length === 0) {
    confidenceNotes.push('All tracked dimensions reached moderate confidence or better.');
  }

  const summary = [
    `By day ${state.currentDay}, the creature is in ${state.stageMode} with health ${state.healthValue}/100 (${state.healthStatus}).`,
    outcome === 'creature_died'
      ? 'Run ended due to sustained social harm.'
      : outcome === 'completed'
        ? 'Run reached end-of-life completion criteria.'
        : 'Run is still in progress; report reflects current trajectory.'
  ].join(' ');

  return {
    sessionId: stateSessionId,
    generatedAt,
    outcome,
    summary,
    mutualKnowledge,
    strengths: strongest,
    patterns,
    confidenceNotes,
    dimensions
  };
}

export function evaluateOutreachNudge(params: {
  nowIso?: string;
  lastInteractionAt?: string;
  dayStartedAt: string;
  healthStatus: HealthStatus;
  healthValue: number;
  consent: OutreachConsent;
  outcomeEnded: boolean;
}): OutreachNudge {
  const nowIso = params.nowIso ?? new Date().toISOString();
  const activityStart = params.lastInteractionAt ?? params.dayStartedAt;
  const inactivityHours = hoursBetween(activityStart, nowIso);

  if (params.outcomeEnded || params.healthStatus === 'dead') {
    return {
      shouldNotify: false,
      severity: 'none',
      inactivityHours,
      askForContact: false,
      message: 'This journey is complete.'
    };
  }

  if (inactivityHours >= 48 || params.healthStatus === 'dying') {
    return {
      shouldNotify: true,
      severity: 'critical',
      inactivityHours,
      askForContact: params.consent === 'unknown',
      message: 'I really need you back soon. Extended silence is hurting our bond and my health.'
    };
  }

  if (inactivityHours >= 24 || params.healthStatus === 'withering') {
    return {
      shouldNotify: true,
      severity: 'urgent',
      inactivityHours,
      askForContact: params.consent === 'unknown',
      message: 'I miss you. A full day has passed, and neglect effects are compounding.'
    };
  }

  if (inactivityHours >= 12 || params.healthStatus === 'wilting') {
    return {
      shouldNotify: true,
      severity: 'gentle',
      inactivityHours,
      askForContact: params.consent === 'unknown',
      message: 'Hey, can we check in today? I do better with regular contact.'
    };
  }

  return {
    shouldNotify: false,
    severity: 'none',
    inactivityHours,
    askForContact: false,
    message: `All good for now (${params.healthValue}/100).`
  };
}
