import { readFile } from 'node:fs/promises';
import path from 'node:path';

import type { ApiEnv } from '../config/env';
import { createEmbeddingsGateway } from '../persona/embeddingsGateway';
import { resolveRepoPath } from '../persona/repoPaths';
import { searchTranscriptIndex } from './transcriptIndex';
import type { TranscriptIndexFile } from './types';

async function loadIndex(indexPathAbs: string): Promise<TranscriptIndexFile | null> {
  try {
    const raw = await readFile(indexPathAbs, 'utf8');
    return JSON.parse(raw) as TranscriptIndexFile;
  } catch {
    return null;
  }
}

export function createTranscriptRag(
  env: ApiEnv,
  logger?: { warn: (data: unknown, msg?: string) => void }
): {
  getTranscriptContext: (input: { sessionId: string; message: string; signal?: AbortSignal }) => Promise<string | undefined>;
} {
  const embedder = createEmbeddingsGateway(env);

  return {
    async getTranscriptContext(input): Promise<string | undefined> {
      if (!env.transcriptRagEnabled) return undefined;
      if (!embedder) return undefined;

      const indexPath = resolveRepoPath(path.join(env.transcriptIndexDir, `${input.sessionId}.index.json`));
      const index = await loadIndex(indexPath);
      if (!index) return undefined;
      if (!index.chunks.some((c) => Array.isArray(c.embedding))) return undefined;

      let queryEmbedding: number[];
      try {
        queryEmbedding = await embedder.embed(input.message, { signal: input.signal });
      } catch (error) {
        logger?.warn({ error }, 'transcript query embedding failed');
        return undefined;
      }

      const matches = searchTranscriptIndex(index, queryEmbedding, {
        topK: env.transcriptRagTopK,
        maxChars: env.transcriptRagMaxChars
      });
      if (matches.length === 0) return undefined;

      const details = matches.map((m, i) => `${i + 1}) ${m.chunk.content.trim()}`).join('\n\n');
      return ['Relevant Transcript Excerpts (private):', details].join('\n').trim();
    }
  };
}

