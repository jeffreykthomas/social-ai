import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { GameService } from '../../domain/gameService';

const appearanceSchema = z.object({
  sessionId: z.string().min(1)
});

export function registerAppearanceRoutes(app: FastifyInstance, gameService: GameService): void {
  app.post('/appearance/render_prompt', async (request, reply) => {
    const body = appearanceSchema.parse(request.body);
    const result = await gameService.renderAppearancePrompt(body.sessionId);

    if (!result) return reply.status(404).send({ error: 'Session not found' });

    return reply.send(result);
  });
}
