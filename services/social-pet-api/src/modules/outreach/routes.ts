import type { OutreachConsent } from '@social-pet/domain';
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { GameService } from '../../domain/gameService';

const consentSchema = z.object({
  consent: z.enum(['unknown', 'granted', 'declined'] as [OutreachConsent, OutreachConsent, OutreachConsent]),
  contactHint: z.string().min(1).max(160).optional()
});

export function registerOutreachRoutes(app: FastifyInstance, gameService: GameService): void {
  app.get('/session/:sessionId/outreach', async (request, reply) => {
    const sessionId = (request.params as { sessionId: string }).sessionId;
    const nudge = await gameService.getOutreachNudge(sessionId);

    if (!nudge) return reply.status(404).send({ error: 'Session not found' });
    return reply.send(nudge);
  });

  app.post('/session/:sessionId/outreach/preferences', async (request, reply) => {
    const sessionId = (request.params as { sessionId: string }).sessionId;
    const body = consentSchema.parse(request.body);
    const state = await gameService.updateOutreachPreference(sessionId, body);

    if (!state) return reply.status(404).send({ error: 'Session not found' });
    return reply.send({ state });
  });

  app.post('/session/:sessionId/outreach/mark-sent', async (request, reply) => {
    const sessionId = (request.params as { sessionId: string }).sessionId;
    const state = await gameService.markOutreachSent(sessionId);

    if (!state) return reply.status(404).send({ error: 'Session not found' });
    return reply.send({ state });
  });
}

