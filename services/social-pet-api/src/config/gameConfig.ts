import { readFileSync } from 'node:fs';

import { resolveRepoPath } from '../persona/repoPaths';

export interface SocialPetGameConfig {
  gameplay: {
    timeline: {
      totalDays: number;
      sessionWindowMinutes: number;
      interactionsPerSessionMin: number;
      interactionsPerSessionMax: number;
      interactionsHardCap: number;
    };
    stageGates: {
      middle_aged: { minDay: number; minInteractions: number };
      wise: { minDay: number; minInteractions: number };
      old: { minDay: number; minInteractions: number };
    };
    acts: {
      trialsAndFrictionStartDay: number;
      integrationStartDay: number;
    };
    trialWindow: {
      startDay: number;
      endDay: number;
      earlyRepairCutoffDay: number;
    };
  };
  arena: {
    defaultMode: 'social_pet' | 'reverie';
    defaultModelCount: number;
    defaultPairingsPerModel: number;
    defaultTurnsPerDay: number;
    teacherLogPath: string;
    summaryPath: string;
  };
}

const DEFAULT_CONFIG: SocialPetGameConfig = {
  gameplay: {
    timeline: {
      totalDays: 14,
      sessionWindowMinutes: 10,
      interactionsPerSessionMin: 3,
      interactionsPerSessionMax: 10,
      interactionsHardCap: 14
    },
    stageGates: {
      middle_aged: { minDay: 4, minInteractions: 9 },
      wise: { minDay: 8, minInteractions: 21 },
      old: { minDay: 12, minInteractions: 33 }
    },
    acts: {
      trialsAndFrictionStartDay: 8,
      integrationStartDay: 12
    },
    trialWindow: {
      startDay: 8,
      endDay: 11,
      earlyRepairCutoffDay: 9
    }
  },
  arena: {
    defaultMode: 'social_pet',
    defaultModelCount: 9,
    defaultPairingsPerModel: 8,
    defaultTurnsPerDay: 6,
    teacherLogPath: 'model/training/arena/teacher.jsonl',
    summaryPath: 'model/training/arena/summary.json'
  }
};

function parsePositiveNumber(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  if (value <= 0) return fallback;
  return value;
}

function parseMode(value: unknown, fallback: 'social_pet' | 'reverie'): 'social_pet' | 'reverie' {
  if (value === 'social_pet' || value === 'reverie') return value;
  return fallback;
}

function parsePath(value: unknown, fallback: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) return fallback;
  return value.trim();
}

function mergeConfig(raw: unknown): SocialPetGameConfig {
  if (!raw || typeof raw !== 'object') {
    return DEFAULT_CONFIG;
  }

  const root = raw as Record<string, unknown>;
  const gameplay = (root.gameplay ?? {}) as Record<string, unknown>;
  const timeline = (gameplay.timeline ?? {}) as Record<string, unknown>;
  const stageGates = (gameplay.stageGates ?? {}) as Record<string, unknown>;
  const middleAged = (stageGates.middle_aged ?? {}) as Record<string, unknown>;
  const wise = (stageGates.wise ?? {}) as Record<string, unknown>;
  const old = (stageGates.old ?? {}) as Record<string, unknown>;
  const acts = (gameplay.acts ?? {}) as Record<string, unknown>;
  const trialWindow = (gameplay.trialWindow ?? {}) as Record<string, unknown>;
  const arena = (root.arena ?? {}) as Record<string, unknown>;

  return {
    gameplay: {
      timeline: {
        totalDays: parsePositiveNumber(timeline.totalDays, DEFAULT_CONFIG.gameplay.timeline.totalDays),
        sessionWindowMinutes: parsePositiveNumber(
          timeline.sessionWindowMinutes,
          DEFAULT_CONFIG.gameplay.timeline.sessionWindowMinutes
        ),
        interactionsPerSessionMin: parsePositiveNumber(
          timeline.interactionsPerSessionMin,
          DEFAULT_CONFIG.gameplay.timeline.interactionsPerSessionMin
        ),
        interactionsPerSessionMax: parsePositiveNumber(
          timeline.interactionsPerSessionMax,
          DEFAULT_CONFIG.gameplay.timeline.interactionsPerSessionMax
        ),
        interactionsHardCap: parsePositiveNumber(
          timeline.interactionsHardCap,
          DEFAULT_CONFIG.gameplay.timeline.interactionsHardCap
        )
      },
      stageGates: {
        middle_aged: {
          minDay: parsePositiveNumber(middleAged.minDay, DEFAULT_CONFIG.gameplay.stageGates.middle_aged.minDay),
          minInteractions: parsePositiveNumber(
            middleAged.minInteractions,
            DEFAULT_CONFIG.gameplay.stageGates.middle_aged.minInteractions
          )
        },
        wise: {
          minDay: parsePositiveNumber(wise.minDay, DEFAULT_CONFIG.gameplay.stageGates.wise.minDay),
          minInteractions: parsePositiveNumber(wise.minInteractions, DEFAULT_CONFIG.gameplay.stageGates.wise.minInteractions)
        },
        old: {
          minDay: parsePositiveNumber(old.minDay, DEFAULT_CONFIG.gameplay.stageGates.old.minDay),
          minInteractions: parsePositiveNumber(old.minInteractions, DEFAULT_CONFIG.gameplay.stageGates.old.minInteractions)
        }
      },
      acts: {
        trialsAndFrictionStartDay: parsePositiveNumber(
          acts.trialsAndFrictionStartDay,
          DEFAULT_CONFIG.gameplay.acts.trialsAndFrictionStartDay
        ),
        integrationStartDay: parsePositiveNumber(acts.integrationStartDay, DEFAULT_CONFIG.gameplay.acts.integrationStartDay)
      },
      trialWindow: {
        startDay: parsePositiveNumber(trialWindow.startDay, DEFAULT_CONFIG.gameplay.trialWindow.startDay),
        endDay: parsePositiveNumber(trialWindow.endDay, DEFAULT_CONFIG.gameplay.trialWindow.endDay),
        earlyRepairCutoffDay: parsePositiveNumber(
          trialWindow.earlyRepairCutoffDay,
          DEFAULT_CONFIG.gameplay.trialWindow.earlyRepairCutoffDay
        )
      }
    },
    arena: {
      defaultMode: parseMode(arena.defaultMode, DEFAULT_CONFIG.arena.defaultMode),
      defaultModelCount: parsePositiveNumber(arena.defaultModelCount, DEFAULT_CONFIG.arena.defaultModelCount),
      defaultPairingsPerModel: parsePositiveNumber(
        arena.defaultPairingsPerModel,
        DEFAULT_CONFIG.arena.defaultPairingsPerModel
      ),
      defaultTurnsPerDay: parsePositiveNumber(arena.defaultTurnsPerDay, DEFAULT_CONFIG.arena.defaultTurnsPerDay),
      teacherLogPath: parsePath(arena.teacherLogPath, DEFAULT_CONFIG.arena.teacherLogPath),
      summaryPath: parsePath(arena.summaryPath, DEFAULT_CONFIG.arena.summaryPath)
    }
  };
}

let cachedConfig: SocialPetGameConfig | null = null;

export function loadSocialPetGameConfig(): SocialPetGameConfig {
  if (cachedConfig) return cachedConfig;

  const configPath = resolveRepoPath(process.env.SOCIAL_PET_GAME_CONFIG_PATH ?? 'config/social-pet-game.json');

  try {
    const raw = JSON.parse(readFileSync(configPath, 'utf8')) as unknown;
    cachedConfig = mergeConfig(raw);
  } catch {
    cachedConfig = DEFAULT_CONFIG;
  }

  return cachedConfig;
}

export function resetSocialPetGameConfigForTests(): void {
  cachedConfig = null;
}
