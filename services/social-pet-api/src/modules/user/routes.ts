import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { GameService } from '../../domain/gameService';
import { loadUserModelDoc } from '../../user/userModelFiles';
import type { UserModelJobRecord } from '../../user/userModelJobManager';

const updateSchema = z.object({
  sessionId: z.string().min(1)
});

export function registerUserRoutes(
  app: FastifyInstance,
  deps: {
    env: { userModelDir: string; userModelEnabled: boolean };
    gameService: Pick<GameService, 'getSession'>;
    userModelJobs: {
      queueUpdateFromSession: (input: {
        userId: string;
        sessionId: string;
        state: any;
        events: any[];
        lastEvent: any;
      }) => UserModelJobRecord;
      getJob: (jobId: string) => UserModelJobRecord | null;
      getLatestForUser: (userId: string) => UserModelJobRecord | null;
    };
  }
): void {
  app.get('/user/:userId/model', async (request, reply) => {
    if (!deps.env.userModelEnabled) return reply.status(404).send({ error: 'user_model_disabled' });
    const userId = (request.params as { userId: string }).userId;
    const doc = await loadUserModelDoc(deps.env.userModelDir, userId);
    if (!doc) return reply.status(404).send({ error: 'user_model_not_found' });
    return reply.send(doc);
  });

  app.post('/user/:userId/model/update', async (request, reply) => {
    if (!deps.env.userModelEnabled) return reply.status(404).send({ error: 'user_model_disabled' });
    const userId = (request.params as { userId: string }).userId;
    const body = updateSchema.parse(request.body);
    const record = await deps.gameService.getSession(body.sessionId);
    if (!record) return reply.status(404).send({ error: 'session_not_found' });
    if (!record.state.userId || record.state.userId !== userId) {
      return reply.status(409).send({ error: 'session_user_mismatch' });
    }

    const lastEvent = record.events[record.events.length - 1];
    if (!lastEvent) return reply.status(409).send({ error: 'no_events' });

    const job = deps.userModelJobs.queueUpdateFromSession({
      userId,
      sessionId: body.sessionId,
      state: record.state,
      events: record.events,
      lastEvent
    });

    return reply.status(202).send({ jobId: job.id, status: job.status });
  });

  app.get('/user/:userId/model/jobs/:jobId', async (request, reply) => {
    const jobId = (request.params as { jobId: string }).jobId;
    const job = deps.userModelJobs.getJob(jobId);
    if (!job) return reply.status(404).send({ error: 'job_not_found' });
    return reply.send(job);
  });

  app.get('/user/:userId/model/jobs/latest', async (request, reply) => {
    const userId = (request.params as { userId: string }).userId;
    const job = deps.userModelJobs.getLatestForUser(userId);
    if (!job) return reply.status(404).send({ error: 'job_not_found' });
    return reply.send(job);
  });
}

