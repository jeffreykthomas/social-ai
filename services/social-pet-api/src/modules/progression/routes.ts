import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { GameService } from '../../domain/gameService';

const tickSchema = z.object({
  sessionId: z.string().min(1)
});

export function registerProgressionRoutes(app: FastifyInstance, gameService: GameService): void {
  app.post('/progression/tick', async (request, reply) => {
    const body = tickSchema.parse(request.body);
    const nextState = await gameService.tickProgression(body.sessionId);

    if (!nextState) return reply.status(404).send({ error: 'Session not found' });

    return reply.send({ state: nextState });
  });
}
