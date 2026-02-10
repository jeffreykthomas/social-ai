import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { GameService } from '../../domain/gameService';

const interactionSchema = z.object({
  sessionId: z.string().min(1),
  message: z.string().min(1)
});

export function registerInteractionRoutes(app: FastifyInstance, gameService: GameService): void {
  app.post('/interaction/respond', async (request, reply) => {
    const body = interactionSchema.parse(request.body);
    try {
      const result = await gameService.respondToInteraction(body.sessionId, body.message);

      if (!result) return reply.status(404).send({ error: 'Session not found' });

      return reply.send(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'interaction_failed';
      if (message === 'session_window_exhausted' || message === 'session_ended' || message === 'timeline_complete') {
        return reply.status(409).send({ error: message });
      }

      throw error;
    }
  });
}
