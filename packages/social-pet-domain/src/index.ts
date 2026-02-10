export type NeedKey = 'connection' | 'safety' | 'approval' | 'empathy' | 'autonomy' | 'fun';

export type SocialNeedState = Record<NeedKey, number>;
export type NeedDeltaState = Partial<Record<NeedKey, number>>;

export type HeroJourneyAct = 'safe_bonding' | 'trials_and_friction' | 'integration';

// Adult-only life-phase progression (kept as 4 stages to preserve existing 4-gate cadence).
export type StageMode = 'young_adult' | 'middle_aged' | 'wise' | 'old';
export type InteractionTone = 'supportive' | 'neutral' | 'harsh';
export type ModelProvider = 'openai' | 'anthropic' | 'heuristic';
export type HealthStatus = 'healthy' | 'wilting' | 'withering' | 'dying' | 'dead';
export type SessionOutcomeReason = 'completed' | 'creature_died' | 'abandoned';
export type OutreachConsent = 'unknown' | 'granted' | 'declined';
export type AssessmentDimensionKey =
  | 'warmth'
  | 'confrontation_comfort'
  | 'autonomy_support'
  | 'emotional_attunement';

export interface AssessmentEvidence {
  eventId: string;
  at: string;
  score: number;
  note: string;
  interactionId?: string;
}

export interface AssessmentDimensionState {
  key: AssessmentDimensionKey;
  score: number;
  confidence: number;
  evidenceCount: number;
  evidence: AssessmentEvidence[];
}

export interface PlayerAssessmentState {
  dimensions: Record<AssessmentDimensionKey, AssessmentDimensionState>;
  totalEvidence: number;
  uncertainDimensions: AssessmentDimensionKey[];
}

export interface CreatureHealthState {
  value: number;
  status: HealthStatus;
  lastDelta: number;
  lastCause: string;
  consecutiveSupportive: number;
  consecutiveHarsh: number;
  consecutiveMissedDays: number;
}

export interface GameTimelineState {
  totalDays: number;
  currentDay: number;
  sessionWindowMinutes: number;
  interactionsPerSessionMin: number;
  interactionsPerSessionMax: number;
  interactionsToday: number;
  sessionsCompleted: number;
  startedAt: string;
  dayStartedAt: string;
  lastInteractionAt?: string;
}

export interface SessionOutcomeState {
  ended: boolean;
  reason?: SessionOutcomeReason;
  endedAt?: string;
}

export interface OutreachState {
  consent: OutreachConsent;
  contactHint?: string;
  lastNudgeAt?: string;
  nudgeCount: number;
}

export interface OutreachNudge {
  shouldNotify: boolean;
  severity: 'none' | 'gentle' | 'urgent' | 'critical';
  inactivityHours: number;
  askForContact: boolean;
  message: string;
}

export interface SessionState {
  sessionId: string;
  userId?: string;
  needs: SocialNeedState;
  stage: {
    mode: StageMode;
  };
  knowledge: MutualKnowledgeState;
  bond: {
    trust: number;
    ruptureRepairBalance: number;
  };
  progress: {
    maturity: number;
    interactions: number;
  };
  narrative: {
    act: HeroJourneyAct;
    activeTrial: string | null;
  };
  timeline: GameTimelineState;
  assessment: PlayerAssessmentState;
  health: CreatureHealthState;
  outcome: SessionOutcomeState;
  outreach: OutreachState;
  latestResponseText?: string;
}

export interface InteractionEvent {
  id: string;
  at: string;
  userMessage: string;
  responseText: string;
  tone: InteractionTone;
  needDeltas: NeedDeltaState;
  provider: ModelProvider;
  model: string;
  providerResponseId?: string;
  sessionDay?: number;
  seedInteractionId?: string;
  assessmentScores?: Partial<Record<AssessmentDimensionKey, number>>;
  assessmentSignals?: string[];
  healthDelta?: number;
  healthStatus?: HealthStatus;
}

export type SeedInteractionStage = StageMode;
export type SeedInteractionKind = 'bonding' | 'conflict' | 'repair' | 'probe' | 'reflection';

export interface SeedInteraction {
  id: string;
  stage: SeedInteractionStage;
  dayRange: [number, number];
  kind: SeedInteractionKind;
  prompt: string;
  measures: AssessmentDimensionKey[];
  tags: string[];
}

export interface RunReportDimensionSummary {
  key: AssessmentDimensionKey;
  score: number;
  confidence: number;
  interpretation: string;
}

export interface SessionRunReport {
  sessionId: string;
  generatedAt: string;
  outcome: SessionOutcomeReason | 'in_progress';
  summary: string;
  mutualKnowledge: MutualKnowledgeReport;
  strengths: string[];
  patterns: string[];
  confidenceNotes: string[];
  dimensions: RunReportDimensionSummary[];
}

export type UserFactCategory =
  | 'identity'
  | 'demographics'
  | 'location'
  | 'work'
  | 'relationship'
  | 'hobby'
  | 'preference'
  | 'goal'
  | 'value'
  | 'other';

// Mirrors the facet categories used in `model/character.json` (curated subset for v0 scoring).
export type CharacterFacetKey =
  | 'motivations_needs_drives'
  | 'values_beliefs_ethics'
  | 'social_style_relationship_patterns'
  | 'emotion_affect_regulation'
  | 'communication_voice_expression'
  | 'attachment_intimacy_boundaries'
  | 'conflict_power_strategy'
  | 'growth_arc_change_dynamics';

export interface UserFact {
  id: string;
  at: string;
  category: UserFactCategory;
  fact: string;
  eventId: string;
}

export interface CharacterFacetReveal {
  id: string;
  at: string;
  facet: CharacterFacetKey;
  evidence: string;
  eventId: string;
}

export interface MutualKnowledgeState {
  points: number;
  userFacts: UserFact[];
  characterFacetsRevealed: CharacterFacetReveal[];
}

export interface MutualKnowledgeReport {
  points: number;
  userFactsCount: number;
  characterFacetsRevealedCount: number;
  topUserFactCategories: Array<{ category: UserFactCategory; count: number }>;
  characterFacetsRevealed: CharacterFacetKey[];
  sampleUserFacts: UserFact[];
}
