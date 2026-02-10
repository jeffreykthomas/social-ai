import { randomUUID } from 'node:crypto';

import type { InteractionEvent, SessionState } from '@social-pet/domain';

import type { ApiEnv } from '../config/env';
import { createUserModelUpdater } from './userModelUpdater';
import type { UserModelDoc } from './types';

export type UserModelJobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export type UserModelJobRecord = {
  id: string;
  userId: string;
  status: UserModelJobStatus;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  error?: string;
  result?: { updatedAt: string };
};

type PendingKey = string;

export function createUserModelJobManager(env: ApiEnv, logger?: { warn: (data: unknown, msg?: string) => void }): {
  queueUpdateFromSession: (input: {
    userId: string;
    sessionId: string;
    state: SessionState;
    events: InteractionEvent[];
    lastEvent: InteractionEvent;
  }) => UserModelJobRecord;
  getJob: (jobId: string) => UserModelJobRecord | null;
  getLatestForUser: (userId: string) => UserModelJobRecord | null;
} {
  const updater = createUserModelUpdater(env, logger);
  const jobs = new Map<string, UserModelJobRecord>();
  const latestByUser = new Map<string, string>();
  const activeByUser = new Map<string, string>();
  const pendingByUser = new Map<PendingKey, { input: Parameters<typeof updater.updateFromSession>[0]; dirty: boolean }>();

  function getJob(jobId: string): UserModelJobRecord | null {
    return jobs.get(jobId) ?? null;
  }

  function getLatestForUser(userId: string): UserModelJobRecord | null {
    const id = latestByUser.get(userId);
    if (!id) return null;
    return jobs.get(id) ?? null;
  }

  function startJob(userId: string): void {
    const pending = pendingByUser.get(userId);
    if (!pending) return;

    const active = activeByUser.get(userId);
    if (active) return;

    const jobId = `ujob_${randomUUID()}`;
    const record: UserModelJobRecord = {
      id: jobId,
      userId,
      status: 'queued',
      createdAt: new Date().toISOString()
    };
    jobs.set(jobId, record);
    latestByUser.set(userId, jobId);
    activeByUser.set(userId, jobId);

    pendingByUser.set(userId, { input: pending.input, dirty: false });

    void (async () => {
      record.status = 'running';
      record.startedAt = new Date().toISOString();

      try {
        const doc: UserModelDoc = await updater.updateFromSession(pending.input);
        record.status = 'succeeded';
        record.finishedAt = new Date().toISOString();
        record.result = { updatedAt: doc.updated_at };
      } catch (error) {
        record.status = 'failed';
        record.finishedAt = new Date().toISOString();
        record.error = error instanceof Error ? error.message : 'user_model_job_failed';
        logger?.warn({ error, jobId: record.id, userId }, 'user model job failed');
      } finally {
        const latestPending = pendingByUser.get(userId);
        const dirty = Boolean(latestPending?.dirty);

        activeByUser.delete(userId);

        if (dirty) {
          // Run once more with the newest input.
          pendingByUser.set(userId, { input: latestPending!.input, dirty: false });
          setTimeout(() => startJob(userId), 250);
        }
      }
    })();
  }

  return {
    queueUpdateFromSession(input) {
      if (!env.userModelEnabled) {
        const record: UserModelJobRecord = {
          id: `ujob_${randomUUID()}`,
          userId: input.userId,
          status: 'failed',
          createdAt: new Date().toISOString(),
          finishedAt: new Date().toISOString(),
          error: 'user_model_disabled'
        };
        jobs.set(record.id, record);
        latestByUser.set(input.userId, record.id);
        return record;
      }

      const existing = pendingByUser.get(input.userId);
      if (existing) {
        existing.input = input;
        existing.dirty = true;
      } else {
        pendingByUser.set(input.userId, { input, dirty: false });
      }

      // If no active job, start one; otherwise it will rerun once after completion.
      startJob(input.userId);

      const latest = getLatestForUser(input.userId);
      if (latest) return latest;

      // Shouldn't happen, but keep type stable.
      const record: UserModelJobRecord = {
        id: `ujob_${randomUUID()}`,
        userId: input.userId,
        status: 'queued',
        createdAt: new Date().toISOString()
      };
      jobs.set(record.id, record);
      latestByUser.set(input.userId, record.id);
      return record;
    },
    getJob,
    getLatestForUser
  };
}

