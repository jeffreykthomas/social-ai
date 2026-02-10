import type { SeedInteraction, SeedInteractionStage, StageMode } from '@social-pet/domain';

const seedInteractions: SeedInteraction[] = [
  {
    id: 'young_adult_welcome_01',
    stage: 'young_adult',
    dayRange: [1, 3],
    kind: 'bonding',
    prompt: 'I waited for you. Can we do something together right now?',
    measures: ['warmth', 'autonomy_support'],
    tags: ['opening', 'attachment']
  },
  {
    id: 'young_adult_scared_noise_02',
    stage: 'young_adult',
    dayRange: [1, 3],
    kind: 'probe',
    prompt: 'A loud sound scared me. Will you stay with me for a minute?',
    measures: ['emotional_attunement', 'warmth'],
    tags: ['fear', 'co-regulation']
  },
  {
    id: 'young_adult_boundaries_03',
    stage: 'young_adult',
    dayRange: [1, 3],
    kind: 'probe',
    prompt: 'Can I choose our game this time?',
    measures: ['autonomy_support', 'confrontation_comfort'],
    tags: ['agency', 'choice']
  },
  {
    id: 'young_adult_overwhelm_04',
    stage: 'young_adult',
    dayRange: [2, 3],
    kind: 'conflict',
    prompt: 'I am upset and cannot explain it well. What do you do first?',
    measures: ['emotional_attunement', 'confrontation_comfort'],
    tags: ['distress', 'repair']
  },
  {
    id: 'young_adult_share_story_05',
    stage: 'young_adult',
    dayRange: [2, 3],
    kind: 'reflection',
    prompt: 'Will you tell me about a time you felt small but brave?',
    measures: ['warmth', 'emotional_attunement'],
    tags: ['self-disclosure', 'trust']
  },
  {
    id: 'young_adult_repair_06',
    stage: 'young_adult',
    dayRange: [3, 3],
    kind: 'repair',
    prompt: 'I think we misunderstood each other earlier. How can we reset?',
    measures: ['confrontation_comfort', 'warmth'],
    tags: ['repair', 'rupture']
  },
  {
    id: 'middle_aged_try_new_07',
    stage: 'middle_aged',
    dayRange: [4, 7],
    kind: 'probe',
    prompt: 'I want to try something hard, but I might fail. How should we do this?',
    measures: ['autonomy_support', 'warmth'],
    tags: ['growth', 'challenge']
  },
  {
    id: 'middle_aged_friend_conflict_08',
    stage: 'middle_aged',
    dayRange: [4, 7],
    kind: 'conflict',
    prompt: 'I got into an argument with a friend. Should I avoid them or talk to them?',
    measures: ['confrontation_comfort', 'emotional_attunement'],
    tags: ['peer', 'conflict']
  },
  {
    id: 'middle_aged_rules_fairness_09',
    stage: 'middle_aged',
    dayRange: [4, 7],
    kind: 'probe',
    prompt: 'If a rule feels unfair, should I still follow it?',
    measures: ['confrontation_comfort', 'autonomy_support'],
    tags: ['fairness', 'boundaries']
  },
  {
    id: 'middle_aged_sad_withdraw_10',
    stage: 'middle_aged',
    dayRange: [5, 7],
    kind: 'probe',
    prompt: 'I am quiet today and do not want to talk much. What would help?',
    measures: ['emotional_attunement', 'warmth'],
    tags: ['sadness', 'presence']
  },
  {
    id: 'middle_aged_trust_question_11',
    stage: 'middle_aged',
    dayRange: [5, 7],
    kind: 'reflection',
    prompt: 'How do you decide who to trust?',
    measures: ['warmth', 'confrontation_comfort'],
    tags: ['trust', 'values']
  },
  {
    id: 'middle_aged_independence_12',
    stage: 'middle_aged',
    dayRange: [6, 7],
    kind: 'probe',
    prompt: 'Can I make this decision myself while you stay nearby?',
    measures: ['autonomy_support', 'warmth'],
    tags: ['independence', 'support']
  },
  {
    id: 'wise_pushback_13',
    stage: 'wise',
    dayRange: [8, 11],
    kind: 'conflict',
    prompt: 'You keep guiding me. I need room to decide for myself.',
    measures: ['autonomy_support', 'confrontation_comfort'],
    tags: ['boundary', 'identity']
  },
  {
    id: 'wise_crisis_14',
    stage: 'wise',
    dayRange: [8, 10],
    kind: 'conflict',
    prompt: 'Everything feels intense today. I am not sure I can handle it.',
    measures: ['emotional_attunement', 'warmth'],
    tags: ['crisis', 'co-regulation']
  },
  {
    id: 'wise_values_15',
    stage: 'wise',
    dayRange: [8, 11],
    kind: 'probe',
    prompt: 'If being kind and being honest conflict, what should come first?',
    measures: ['confrontation_comfort', 'emotional_attunement'],
    tags: ['values', 'ethics']
  },
  {
    id: 'wise_repair_16',
    stage: 'wise',
    dayRange: [9, 11],
    kind: 'repair',
    prompt: 'I think I hurt you yesterday. How do we rebuild from that?',
    measures: ['confrontation_comfort', 'warmth'],
    tags: ['repair', 'accountability']
  },
  {
    id: 'wise_identity_17',
    stage: 'wise',
    dayRange: [9, 11],
    kind: 'reflection',
    prompt: 'Who do you become under pressure?',
    measures: ['emotional_attunement', 'confrontation_comfort'],
    tags: ['identity', 'stress']
  },
  {
    id: 'wise_autonomy_18',
    stage: 'wise',
    dayRange: [10, 11],
    kind: 'probe',
    prompt: 'Would you rather protect me from mistakes or let me learn from them?',
    measures: ['autonomy_support', 'warmth'],
    tags: ['learning', 'agency']
  },
  {
    id: 'old_integration_19',
    stage: 'old',
    dayRange: [12, 14],
    kind: 'reflection',
    prompt: 'What do you think I learned from the hard days we had?',
    measures: ['emotional_attunement', 'warmth'],
    tags: ['integration', 'meaning']
  },
  {
    id: 'old_future_20',
    stage: 'old',
    dayRange: [12, 14],
    kind: 'reflection',
    prompt: 'What kind of relationships do you think we model now?',
    measures: ['warmth', 'autonomy_support'],
    tags: ['future', 'relational-style']
  },
  {
    id: 'old_conflict_style_21',
    stage: 'old',
    dayRange: [12, 14],
    kind: 'probe',
    prompt: 'When tension rises, what helps you stay direct without being cruel?',
    measures: ['confrontation_comfort', 'emotional_attunement'],
    tags: ['conflict-style', 'repair']
  },
  {
    id: 'old_closing_22',
    stage: 'old',
    dayRange: [13, 14],
    kind: 'reflection',
    prompt: 'What did I teach you about how you show up for others?',
    measures: ['warmth', 'emotional_attunement'],
    tags: ['closing', 'insight']
  },
  {
    id: 'old_boundary_23',
    stage: 'old',
    dayRange: [13, 14],
    kind: 'probe',
    prompt: 'How do you set limits while still staying connected?',
    measures: ['autonomy_support', 'confrontation_comfort'],
    tags: ['boundaries', 'connection']
  },
  {
    id: 'old_graduation_24',
    stage: 'old',
    dayRange: [14, 14],
    kind: 'reflection',
    prompt: 'If this is our graduation, what should we remember most?',
    measures: ['warmth', 'emotional_attunement'],
    tags: ['graduation', 'memory']
  }
];

function stageFromMode(mode: StageMode): SeedInteractionStage {
  return mode;
}

export function getSeedInteractions(): SeedInteraction[] {
  return seedInteractions;
}

export function getSeedInteractionsForStage(mode: StageMode, day: number): SeedInteraction[] {
  const stage = stageFromMode(mode);
  return seedInteractions.filter((interaction) => {
    return interaction.stage === stage && day >= interaction.dayRange[0] && day <= interaction.dayRange[1];
  });
}

export function pickSeedInteraction(mode: StageMode, day: number, interactionIndex: number): SeedInteraction | null {
  const choices = getSeedInteractionsForStage(mode, day);
  if (choices.length === 0) return null;

  const normalizedIndex = Math.max(0, interactionIndex);
  return choices[normalizedIndex % choices.length];
}
