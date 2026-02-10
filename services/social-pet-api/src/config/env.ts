import type { ModelProvider } from '@social-pet/domain';

export type PersistenceMode = 'memory' | 'redis' | 'postgres' | 'hybrid';

export interface ApiEnv {
  port: number;
  corsOrigins: string[];
  llmProvider: ModelProvider;
  llmTimeoutMs: number;
  llmHistoryTurns: number;
  openaiApiKey?: string;
  openaiModel: string;
  openaiEmbeddingModel: string;
  anthropicApiKey?: string;
  anthropicModel: string;
  persistenceMode: PersistenceMode;
  databaseUrl?: string;
  redisUrl?: string;
  eventLogMax: number;
  personaRagEnabled: boolean;
  personaDocPath: string;
  personaIndexPath: string;
  personaRagTopK: number;
  personaRagMaxChars: number;
  transcriptEnabled: boolean;
  transcriptRagEnabled: boolean;
  transcriptDir: string;
  transcriptIndexDir: string;
  transcriptRagTopK: number;
  transcriptRagMaxChars: number;
  userModelEnabled: boolean;
  userModelUpdateOnInteraction: boolean;
  userModelDir: string;
  userModelOpenAIModel: string;
  elevenlabsApiKey?: string;
  elevenlabsVoiceIdsFemale: string[];
  elevenlabsVoiceIdsMale: string[];
  elevenlabsModelId: string;
  elevenlabsAgentId?: string;
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  const normalized = value.trim().toLowerCase();
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'y' || normalized === 'on') {
    return true;
  }
  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'n' || normalized === 'off') {
    return false;
  }
  return fallback;
}

function parseProvider(value: string | undefined): ModelProvider {
  if (value === 'openai' || value === 'anthropic' || value === 'heuristic') {
    return value;
  }
  return 'heuristic';
}

function parsePersistenceMode(value: string | undefined): PersistenceMode {
  if (value === 'memory' || value === 'redis' || value === 'postgres' || value === 'hybrid') {
    return value;
  }
  return 'memory';
}

function parseCsv(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}


export function loadEnv(): ApiEnv {
  return {
    port: parseNumber(process.env.PORT, 3001),
    corsOrigins: parseCsv(process.env.CORS_ORIGINS),
    llmProvider: parseProvider(process.env.LLM_PROVIDER),
    llmTimeoutMs: parseNumber(process.env.LLM_TIMEOUT_MS, 900),
    llmHistoryTurns: parseNumber(process.env.LLM_HISTORY_TURNS, 8),
    openaiApiKey: process.env.OPENAI_API_KEY,
    openaiModel: process.env.OPENAI_MODEL ?? 'gpt-4.1-mini',
    openaiEmbeddingModel: process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-3-small',
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    anthropicModel: process.env.ANTHROPIC_MODEL ?? 'claude-3-5-haiku-latest',
    persistenceMode: parsePersistenceMode(process.env.PERSISTENCE_MODE),
    databaseUrl: process.env.DATABASE_URL,
    redisUrl: process.env.REDIS_URL,
    eventLogMax: parseNumber(process.env.EVENT_LOG_MAX, 200),
    personaRagEnabled: parseBoolean(process.env.PERSONA_RAG_ENABLED, false),
    personaDocPath: process.env.PERSONA_DOC_PATH ?? 'model/persona.generated.json',
    personaIndexPath: process.env.PERSONA_INDEX_PATH ?? 'model/persona.generated.index.json',
    personaRagTopK: parseNumber(process.env.PERSONA_RAG_TOP_K, 4),
    personaRagMaxChars: parseNumber(process.env.PERSONA_RAG_MAX_CHARS, 1400),
    transcriptEnabled: parseBoolean(process.env.TRANSCRIPT_ENABLED, true),
    transcriptRagEnabled: parseBoolean(process.env.TRANSCRIPT_RAG_ENABLED, false),
    transcriptDir: process.env.TRANSCRIPT_DIR ?? 'model/transcripts',
    transcriptIndexDir: process.env.TRANSCRIPT_INDEX_DIR ?? 'model/transcripts-index',
    transcriptRagTopK: parseNumber(process.env.TRANSCRIPT_RAG_TOP_K, 4),
    transcriptRagMaxChars: parseNumber(process.env.TRANSCRIPT_RAG_MAX_CHARS, 1400),
    userModelEnabled: parseBoolean(process.env.USER_MODEL_ENABLED, false),
    userModelUpdateOnInteraction: parseBoolean(process.env.USER_MODEL_UPDATE_ON_INTERACTION, false),
    userModelDir: process.env.USER_MODEL_DIR ?? 'model/users',
    userModelOpenAIModel: process.env.USER_MODEL_OPENAI_MODEL ?? 'gpt-5.2',
    elevenlabsApiKey: process.env.ELEVENLABS_API_KEY,
    elevenlabsVoiceIdsFemale: parseCsv(process.env.ELEVENLABS_VOICE_IDS_FEMALE),
    elevenlabsVoiceIdsMale: parseCsv(process.env.ELEVENLABS_VOICE_IDS_MALE),
    elevenlabsModelId: process.env.ELEVENLABS_MODEL_ID ?? 'eleven_turbo_v2_5',
    elevenlabsAgentId: process.env.ELEVENLABS_AGENT_ID
  };
}
