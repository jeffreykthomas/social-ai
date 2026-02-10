import { readFile } from 'node:fs/promises';
import path from 'node:path';

import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { ApiEnv } from '../../config/env';
import { createEmbeddingsGateway } from '../../persona/embeddingsGateway';
import { resolveRepoPath } from '../../persona/repoPaths';
import { searchTranscriptIndex } from '../../transcript/transcriptIndex';
import type { TranscriptIndexFile } from '../../transcript/types';

const searchSchema = z.object({
  query: z.string().min(1),
  topK: z.coerce.number().int().min(1).max(12).optional()
});

async function loadIndex(indexPathAbs: string): Promise<TranscriptIndexFile | null> {
  try {
    const raw = await readFile(indexPathAbs, 'utf8');
    return JSON.parse(raw) as TranscriptIndexFile;
  } catch {
    return null;
  }
}

export function registerTranscriptRoutes(app: FastifyInstance, env: ApiEnv): void {
  const embedder = createEmbeddingsGateway(env);

  app.get('/transcript/:sessionId', async (request, reply) => {
    if (!env.transcriptEnabled) return reply.status(404).send({ error: 'transcript_disabled' });

    const sessionId = (request.params as { sessionId: string }).sessionId;
    const transcriptPath = resolveRepoPath(path.join(env.transcriptDir, `${sessionId}.jsonl`));

    try {
      const raw = await readFile(transcriptPath, 'utf8');
      return reply.send({ sessionId, transcript: raw });
    } catch {
      return reply.status(404).send({ error: 'transcript_not_found' });
    }
  });

  app.post('/transcript/:sessionId/search', async (request, reply) => {
    if (!env.transcriptEnabled) return reply.status(404).send({ error: 'transcript_disabled' });
    if (!embedder) return reply.status(409).send({ error: 'openai_api_key_missing' });

    const sessionId = (request.params as { sessionId: string }).sessionId;
    const body = searchSchema.parse(request.body);

    const indexPath = resolveRepoPath(path.join(env.transcriptIndexDir, `${sessionId}.index.json`));
    const index = await loadIndex(indexPath);
    if (!index) return reply.status(404).send({ error: 'transcript_index_not_found' });

    const embedding = await embedder.embed(body.query);
    const matches = searchTranscriptIndex(index, embedding, {
      topK: body.topK ?? env.transcriptRagTopK,
      maxChars: env.transcriptRagMaxChars
    });

    return reply.send({
      sessionId,
      matches: matches.map((m) => ({
        score: m.score,
        at: m.chunk.at,
        eventId: m.chunk.event_id,
        content: m.chunk.content
      }))
    });
  });
}

