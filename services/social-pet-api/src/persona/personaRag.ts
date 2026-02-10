import type { ApiEnv } from '../config/env';
import { createEmbeddingsGateway } from './embeddingsGateway';
import { loadPersonaDocFromPath, loadPersonaIndexFromPath } from './personaFiles';
import { searchIndex } from './personaIndex';
import type { GeneratedPersonaDoc, PersonaIndexFile } from './types';
import { resolveRepoPath } from './repoPaths';

type Cache = {
  loadedAtMs: number;
  doc: GeneratedPersonaDoc;
  index: PersonaIndexFile | null;
};

export function createPersonaRag(
  env: ApiEnv,
  logger?: { warn: (data: unknown, msg?: string) => void }
): {
  getPersonaContext: (input: { message: string; signal?: AbortSignal }) => Promise<string | undefined>;
  getActivePersonaSummary: () => Promise<{ personaId: string; generatedAt: string; promptSummary: string; quickSummary: string } | null>;
} {
  const embedder = createEmbeddingsGateway(env);
  let cache: Cache | null = null;

  async function loadCached(): Promise<Cache | null> {
    const now = Date.now();
    if (cache && now - cache.loadedAtMs < 30_000) return cache;

    try {
      const docPath = resolveRepoPath(env.personaDocPath);
      const indexPath = resolveRepoPath(env.personaIndexPath);

      const doc = await loadPersonaDocFromPath(docPath);
      let index: PersonaIndexFile | null = null;
      try {
        index = await loadPersonaIndexFromPath(indexPath);
      } catch {
        index = null;
      }

      cache = { loadedAtMs: now, doc, index };
      return cache;
    } catch (error) {
      logger?.warn({ error, personaDocPath: env.personaDocPath }, 'persona doc load failed');
      cache = null;
      return null;
    }
  }

  return {
    async getPersonaContext(input): Promise<string | undefined> {
      if (!env.personaRagEnabled) return undefined;

      const loaded = await loadCached();
      if (!loaded) return undefined;

      const { doc, index } = loaded;

      // Always include the compact prompt summary, even if embeddings/index are missing.
      const base = [
        'Character Prompt Summary (private):',
        doc.prompt_summary.trim(),
        ''
      ];

      if (!embedder) {
        return base.join('\n').trim();
      }

      if (!index || !index.chunks.some((c) => Array.isArray(c.embedding))) {
        return base.join('\n').trim();
      }

      const queryEmbedding = await embedder.embed(input.message, { signal: input.signal });
      const matches = searchIndex(index, queryEmbedding, {
        topK: env.personaRagTopK,
        maxChars: env.personaRagMaxChars
      });

      if (matches.length === 0) return base.join('\n').trim();

      const details = matches
        .map((m, i) => `${i + 1}) ${m.chunk.content.trim()}`)
        .join('\n\n');

      return [...base, 'Relevant Persona Details (private):', details].join('\n').trim();
    },

    async getActivePersonaSummary() {
      const loaded = await loadCached();
      if (!loaded) return null;

      // Extract gender from persona demographics (used for voice selection)
      const demographics = loaded.doc.persona?.demographics_social_position as
        | { gender_identity?: string; pronouns?: string }
        | undefined;
      const gender = demographics?.gender_identity ?? demographics?.pronouns ?? undefined;

      return {
        personaId: loaded.doc.persona_id,
        generatedAt: loaded.doc.generated_at,
        promptSummary: loaded.doc.prompt_summary,
        quickSummary: loaded.doc.quick_summary,
        gender
      };
    }
  };
}
