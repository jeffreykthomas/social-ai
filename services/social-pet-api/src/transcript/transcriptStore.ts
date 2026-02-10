import { mkdir, readFile, writeFile, appendFile } from 'node:fs/promises';
import path from 'node:path';

import type { ApiEnv } from '../config/env';
import { createEmbeddingsGateway } from '../persona/embeddingsGateway';
import { resolveRepoPath } from '../persona/repoPaths';
import type { TranscriptIndexFile } from './types';
import { buildTranscriptIndex, makeChunk } from './transcriptIndex';

import type { InteractionEvent } from '@social-pet/domain';

async function loadIndex(pathAbs: string): Promise<TranscriptIndexFile | null> {
  try {
    const raw = await readFile(pathAbs, 'utf8');
    return JSON.parse(raw) as TranscriptIndexFile;
  } catch {
    return null;
  }
}

async function writeJsonPretty(pathAbs: string, data: unknown): Promise<void> {
  await mkdir(path.dirname(pathAbs), { recursive: true });
  await writeFile(pathAbs, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

export function createTranscriptStore(
  env: ApiEnv,
  logger?: { warn: (data: unknown, msg?: string) => void }
): {
  appendInteraction: (sessionId: string, event: InteractionEvent) => Promise<void>;
} {
  const embedder = createEmbeddingsGateway(env);

  return {
    async appendInteraction(sessionId: string, event: InteractionEvent): Promise<void> {
      if (!env.transcriptEnabled) return;

      const transcriptPath = resolveRepoPath(path.join(env.transcriptDir, `${sessionId}.jsonl`));
      const indexPath = resolveRepoPath(path.join(env.transcriptIndexDir, `${sessionId}.index.json`));

      try {
        await mkdir(path.dirname(transcriptPath), { recursive: true });

        // Append JSONL lines (full run transcript).
        const userLine = JSON.stringify({ at: event.at, eventId: event.id, role: 'user', content: event.userMessage }) + '\n';
        const assistantLine =
          JSON.stringify({ at: event.at, eventId: event.id, role: 'assistant', content: event.responseText }) + '\n';
        await appendFile(transcriptPath, userLine + assistantLine, 'utf8');

        // Update the vector index for retrieval (per interaction chunk).
        const existing = (await loadIndex(indexPath)) ?? buildTranscriptIndex({ sessionId, embeddingModel: env.openaiEmbeddingModel });
        const content = `User: ${event.userMessage}\nAssistant: ${event.responseText}`.trim();

        let embedding: number[] | undefined;
        if (embedder) {
          try {
            embedding = await embedder.embed(content);
          } catch (error) {
            logger?.warn({ error }, 'transcript embedding failed; continuing without embedding');
          }
        }

        const chunk = makeChunk({
          chunkIndex: existing.chunks.length,
          at: event.at,
          eventId: event.id,
          content,
          embedding
        });

        existing.chunks.push(chunk);
        await writeJsonPretty(indexPath, existing);
      } catch (error) {
        logger?.warn({ error, sessionId }, 'transcript append failed');
      }
    }
  };
}
