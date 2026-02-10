import { randomUUID } from 'node:crypto';

import type { PersonaIndexChunk, PersonaIndexFile } from './types';
import { cosineSimilarity } from './vector';

export function chunkText(text: string, opts?: { maxChars?: number }): string[] {
  const maxChars = opts?.maxChars ?? 900;
  const paragraphs = text
    .split(/\n{2,}/g)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const chunks: string[] = [];
  let current = '';

  for (const p of paragraphs) {
    const candidate = current.length === 0 ? p : `${current}\n\n${p}`;
    if (candidate.length <= maxChars) {
      current = candidate;
      continue;
    }

    if (current.length > 0) {
      chunks.push(current);
      current = '';
    }

    if (p.length <= maxChars) {
      current = p;
      continue;
    }

    // Hard-split very long paragraphs.
    for (let i = 0; i < p.length; i += maxChars) {
      chunks.push(p.slice(i, i + maxChars));
    }
  }

  if (current.length > 0) chunks.push(current);
  return chunks;
}

export function buildIndexFile(params: {
  personaId: string;
  chunkContents: string[];
  embeddingModel?: string;
  embeddings?: Array<number[] | undefined>;
}): PersonaIndexFile {
  const { personaId, chunkContents, embeddingModel, embeddings } = params;

  const chunks: PersonaIndexChunk[] = chunkContents.map((content, i) => {
    const embedding = embeddings?.[i];
    return {
      id: `pchunk_${randomUUID()}`,
      chunk_index: i,
      content,
      ...(embedding ? { embedding } : {})
    };
  });

  return {
    persona_id: personaId,
    created_at: new Date().toISOString(),
    ...(embeddingModel ? { embedding_model: embeddingModel } : {}),
    chunks
  };
}

export function searchIndex(
  index: PersonaIndexFile,
  queryEmbedding: number[],
  opts?: { topK?: number; maxChars?: number }
): Array<{ chunk: PersonaIndexChunk; score: number }> {
  const topK = opts?.topK ?? 4;
  const maxChars = opts?.maxChars ?? 1400;

  const scored = index.chunks
    .filter((c) => Array.isArray(c.embedding))
    .map((c) => ({ chunk: c, score: cosineSimilarity(queryEmbedding, c.embedding as number[]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, topK));

  // Keep the final payload bounded by chars; drop tail chunks as needed.
  let budget = maxChars;
  const trimmed: Array<{ chunk: PersonaIndexChunk; score: number }> = [];
  for (const entry of scored) {
    const len = entry.chunk.content.length;
    if (budget <= 0) break;
    if (len > budget && trimmed.length > 0) break;
    trimmed.push(entry);
    budget -= len;
  }

  return trimmed;
}

