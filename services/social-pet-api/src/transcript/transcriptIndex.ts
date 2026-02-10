import { randomUUID } from 'node:crypto';

import type { TranscriptIndexChunk, TranscriptIndexFile } from './types';
import { cosineSimilarity } from '../persona/vector';

export function buildTranscriptIndex(params: {
  sessionId: string;
  embeddingModel?: string;
  chunks?: TranscriptIndexChunk[];
}): TranscriptIndexFile {
  return {
    session_id: params.sessionId,
    created_at: new Date().toISOString(),
    ...(params.embeddingModel ? { embedding_model: params.embeddingModel } : {}),
    chunks: params.chunks ?? []
  };
}

export function makeChunk(params: {
  chunkIndex: number;
  at: string;
  eventId: string;
  content: string;
  embedding?: number[];
}): TranscriptIndexChunk {
  return {
    id: `tchunk_${randomUUID()}`,
    chunk_index: params.chunkIndex,
    at: params.at,
    event_id: params.eventId,
    content: params.content,
    ...(params.embedding ? { embedding: params.embedding } : {})
  };
}

export function searchTranscriptIndex(
  index: TranscriptIndexFile,
  queryEmbedding: number[],
  opts?: { topK?: number; maxChars?: number }
): Array<{ chunk: TranscriptIndexChunk; score: number }> {
  const topK = opts?.topK ?? 4;
  const maxChars = opts?.maxChars ?? 1400;

  const scored = index.chunks
    .filter((c) => Array.isArray(c.embedding))
    .map((c) => ({ chunk: c, score: cosineSimilarity(queryEmbedding, c.embedding as number[]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, topK));

  let budget = maxChars;
  const trimmed: Array<{ chunk: TranscriptIndexChunk; score: number }> = [];
  for (const entry of scored) {
    const len = entry.chunk.content.length;
    if (budget <= 0) break;
    if (len > budget && trimmed.length > 0) break;
    trimmed.push(entry);
    budget -= len;
  }

  return trimmed;
}

