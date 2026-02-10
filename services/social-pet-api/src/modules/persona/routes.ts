import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { PersonaJobRecord } from '../../persona/personaJobManager';

const cycleSchema = z
  .object({
    personaId: z.string().min(1).optional(),
    iterations: z.coerce.number().int().min(0).max(6).optional()
  })
  .optional();

export function registerPersonaRoutes(
  app: FastifyInstance,
  deps: {
    personaJobs: {
      startCycle: (params?: { personaId?: string; iterations?: number }) => PersonaJobRecord;
      getJob: (jobId: string) => PersonaJobRecord | null;
      getActiveJob: () => PersonaJobRecord | null;
    };
    personaRag: {
      getActivePersonaSummary: () => Promise<{ personaId: string; generatedAt: string; promptSummary: string; quickSummary: string } | null>;
    };
  }
): void {
  app.get('/persona/active', async (_request, reply) => {
    const summary = await deps.personaRag.getActivePersonaSummary();
    const activeJob = deps.personaJobs.getActiveJob();
    return reply.send({ summary, activeJob });
  });

  app.post('/persona/cycle', async (request, reply) => {
    const body = cycleSchema ? cycleSchema.parse(request.body) : undefined;
    const job = deps.personaJobs.startCycle(body ?? undefined);
    return reply.status(202).send({ jobId: job.id, status: job.status });
  });

  app.get('/persona/jobs/:jobId', async (request, reply) => {
    const jobId = (request.params as { jobId: string }).jobId;
    const job = deps.personaJobs.getJob(jobId);
    if (!job) return reply.status(404).send({ error: 'job_not_found' });
    return reply.send(job);
  });
}

