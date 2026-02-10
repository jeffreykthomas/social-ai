import OpenAI from 'openai';

import type { ApiEnv } from '../config/env';

export type EmbeddingsGateway = {
  embed: (input: string, options?: { signal?: AbortSignal }) => Promise<number[]>;
};

export function createEmbeddingsGateway(env: ApiEnv): EmbeddingsGateway | null {
  if (!env.openaiApiKey) return null;
  const openai = new OpenAI({ apiKey: env.openaiApiKey });

  return {
    async embed(input: string, options?: { signal?: AbortSignal }): Promise<number[]> {
      const resp = await openai.embeddings.create(
        {
          model: env.openaiEmbeddingModel,
          input
        },
        options?.signal ? { signal: options.signal } : undefined
      );

      const embedding = resp.data?.[0]?.embedding;
      if (!embedding) throw new Error('openai_embedding_empty');
      return embedding;
    }
  };
}

