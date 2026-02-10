import type { InteractionEvent, SessionState } from '@social-pet/domain';
import Redis from 'ioredis';
import { Pool } from 'pg';

import type { ApiEnv } from '../config/env';

export type SessionRecord = {
  state: SessionState;
  events: InteractionEvent[];
};

export interface SessionPersistence {
  init: () => Promise<void>;
  load: (sessionId: string) => Promise<SessionRecord | null>;
  upsert: (sessionId: string, state: SessionState, events: InteractionEvent[]) => Promise<void>;
  appendEvent: (sessionId: string, event: InteractionEvent) => Promise<void>;
}

const redisKey = (sessionId: string) => `socialpet:session:${sessionId}`;

function shouldUseRedis(env: ApiEnv): boolean {
  return (env.persistenceMode === 'redis' || env.persistenceMode === 'hybrid') && Boolean(env.redisUrl);
}

function shouldUsePostgres(env: ApiEnv): boolean {
  return (env.persistenceMode === 'postgres' || env.persistenceMode === 'hybrid') && Boolean(env.databaseUrl);
}

export function createSessionPersistence(env: ApiEnv): SessionPersistence {
  const redis = shouldUseRedis(env) ? new Redis(env.redisUrl as string, { maxRetriesPerRequest: 1 }) : null;
  const pg = shouldUsePostgres(env) ? new Pool({ connectionString: env.databaseUrl }) : null;

  async function init(): Promise<void> {
    if (!pg) return;

    await pg.query(`
      CREATE TABLE IF NOT EXISTS social_pet_sessions (
        session_id TEXT PRIMARY KEY,
        state JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
    `);

    await pg.query(`
      CREATE TABLE IF NOT EXISTS social_pet_events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES social_pet_sessions(session_id) ON DELETE CASCADE,
        at TIMESTAMPTZ NOT NULL,
        payload JSONB NOT NULL
      );
    `);

    await pg.query(`
      CREATE INDEX IF NOT EXISTS social_pet_events_session_at_idx
      ON social_pet_events(session_id, at DESC);
    `);
  }

  async function loadFromRedis(sessionId: string): Promise<SessionRecord | null> {
    if (!redis) return null;
    const raw = await redis.get(redisKey(sessionId));
    if (!raw) return null;

    const parsed = JSON.parse(raw) as SessionRecord;
    return parsed;
  }

  async function loadFromPostgres(sessionId: string): Promise<SessionRecord | null> {
    if (!pg) return null;

    const sessionResult = await pg.query<{ state: SessionState }>(
      'SELECT state FROM social_pet_sessions WHERE session_id = $1',
      [sessionId]
    );

    if (sessionResult.rowCount === 0) return null;

    const eventsResult = await pg.query<{ payload: InteractionEvent }>(
      'SELECT payload FROM social_pet_events WHERE session_id = $1 ORDER BY at ASC LIMIT $2',
      [sessionId, env.eventLogMax]
    );

    return {
      state: sessionResult.rows[0].state,
      events: eventsResult.rows.map((row) => row.payload)
    };
  }

  async function persistToRedis(sessionId: string, record: SessionRecord): Promise<void> {
    if (!redis) return;
    await redis.set(redisKey(sessionId), JSON.stringify(record));
  }

  async function persistStateToPostgres(sessionId: string, state: SessionState): Promise<void> {
    if (!pg) return;

    await pg.query(
      `
      INSERT INTO social_pet_sessions(session_id, state)
      VALUES ($1, $2::jsonb)
      ON CONFLICT (session_id)
      DO UPDATE SET state = EXCLUDED.state, updated_at = NOW()
      `,
      [sessionId, JSON.stringify(state)]
    );
  }

  async function persistEventToPostgres(sessionId: string, event: InteractionEvent): Promise<void> {
    if (!pg) return;

    await pg.query(
      `
      INSERT INTO social_pet_events(id, session_id, at, payload)
      VALUES ($1, $2, $3, $4::jsonb)
      ON CONFLICT (id) DO NOTHING
      `,
      [event.id, sessionId, event.at, JSON.stringify(event)]
    );
  }

  return {
    init,

    async load(sessionId: string): Promise<SessionRecord | null> {
      const redisRecord = await loadFromRedis(sessionId);
      if (redisRecord) return redisRecord;

      const pgRecord = await loadFromPostgres(sessionId);
      if (!pgRecord) return null;

      await persistToRedis(sessionId, pgRecord);
      return pgRecord;
    },

    async upsert(sessionId: string, state: SessionState, events: InteractionEvent[]): Promise<void> {
      const trimmed = events.slice(-env.eventLogMax);
      await Promise.all([
        persistToRedis(sessionId, { state, events: trimmed }),
        persistStateToPostgres(sessionId, state)
      ]);
    },

    async appendEvent(sessionId: string, event: InteractionEvent): Promise<void> {
      await persistEventToPostgres(sessionId, event);
    }
  };
}
