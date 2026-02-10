import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { GameService } from '../../domain/gameService';

const startSchema = z.object({
  userId: z.string().min(1).optional(),
  createPersona: z.boolean().optional()
});

const seedQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(12).optional()
});

export function registerSessionRoutes(
  app: FastifyInstance,
  gameService: GameService,
  opts?: { personaJobs?: { startCycle: (params?: { personaId?: string; iterations?: number }) => { id: string } } }
): void {
  app.post('/session/start', async (request, reply) => {
    const body = startSchema.parse(request.body);

    const personaJobId =
      body.createPersona && opts?.personaJobs ? opts.personaJobs.startCycle({ personaId: 'default', iterations: 2 }).id : undefined;
    const result = await gameService.startSession({ userId: body.userId });
    return reply.send({ ...result, meta: { ...(personaJobId ? { personaJobId } : {}) } });
  });

  app.get('/session/:sessionId/state', async (request, reply) => {
    const sessionId = (request.params as { sessionId: string }).sessionId;
    const record = await gameService.getSession(sessionId);

    if (!record) return reply.status(404).send({ error: 'Session not found' });

    return reply.send({ state: record.state, events: record.events });
  });

  app.get('/session/:sessionId/seed-interactions', async (request, reply) => {
    const sessionId = (request.params as { sessionId: string }).sessionId;
    const query = seedQuerySchema.parse(request.query);
    const result = await gameService.getSeedInteractions(sessionId, query.limit);

    if (!result) return reply.status(404).send({ error: 'Session not found' });
    return reply.send({ interactions: result });
  });

  app.get('/session/:sessionId/report', async (request, reply) => {
    const sessionId = (request.params as { sessionId: string }).sessionId;
    const report = await gameService.generateRunReport(sessionId);

    if (!report) return reply.status(404).send({ error: 'Session not found' });
    return reply.send(report);
  });
}
