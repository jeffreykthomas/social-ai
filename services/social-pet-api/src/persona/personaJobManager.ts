import { randomUUID } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import type { ApiEnv } from '../config/env';
import { loadCharacterSchemaFromPath } from './loadCharacterSchema';
import { generatePersonaAdversarial } from './adversarialPersonaGenerator';
import { createEmbeddingsGateway } from './embeddingsGateway';
import { renderPersonaMarkdown } from './renderPersona';
import { buildIndexFile, chunkText } from './personaIndex';
import { pickDefaultPersonaModels } from './personaModelClients';
import { resolveRepoPath } from './repoPaths';

export type PersonaJobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export type PersonaJobRecord = {
  id: string;
  status: PersonaJobStatus;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  error?: string;
  result?: {
    personaId: string;
    docPath: string;
    indexPath: string;
    validationIssueCount: number;
  };
};

async function writeJsonPretty(filePath: string, data: unknown): Promise<void> {
  const abs = resolveRepoPath(filePath);
  await mkdir(path.dirname(abs), { recursive: true });
  await writeFile(abs, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

export function createPersonaJobManager(env: ApiEnv, logger?: { warn: (data: unknown, msg?: string) => void }): {
  startCycle: (params?: { personaId?: string; iterations?: number }) => PersonaJobRecord;
  getJob: (jobId: string) => PersonaJobRecord | null;
  getActiveJob: () => PersonaJobRecord | null;
} {
  const jobs = new Map<string, PersonaJobRecord>();
  let activeJobId: string | null = null;

  function getJob(jobId: string): PersonaJobRecord | null {
    return jobs.get(jobId) ?? null;
  }

  function getActiveJob(): PersonaJobRecord | null {
    if (!activeJobId) return null;
    return jobs.get(activeJobId) ?? null;
  }

  function startCycle(params?: { personaId?: string; iterations?: number }): PersonaJobRecord {
    const existing = getActiveJob();
    if (existing && (existing.status === 'queued' || existing.status === 'running')) {
      return existing;
    }

    const jobId = `pjob_${randomUUID()}`;
    const record: PersonaJobRecord = {
      id: jobId,
      status: 'queued',
      createdAt: new Date().toISOString()
    };
    jobs.set(jobId, record);
    activeJobId = jobId;

    const personaId = params?.personaId ?? 'default';
    const iterations = params?.iterations ?? 2;

    // Fire-and-forget background job.
    void (async () => {
      record.status = 'running';
      record.startedAt = new Date().toISOString();

      try {
        if (!env.openaiApiKey && !env.anthropicApiKey) {
          throw new Error('persona_generation_requires_api_key');
        }

        const schemaPath = resolveRepoPath('model/character.json');
        const schema = await loadCharacterSchemaFromPath(schemaPath);
        const models = pickDefaultPersonaModels(env);

        const { doc, validationIssues } = await generatePersonaAdversarial({
          env,
          schema,
          personaId,
          generator: models.generator,
          critic: models.critic,
          iterations
        });

        await writeJsonPretty(env.personaDocPath, doc);

        const markdown = renderPersonaMarkdown(schema, doc);
        const chunkContents = chunkText(markdown, { maxChars: 900 });

        const embedder = createEmbeddingsGateway(env);
        let embeddings: Array<number[] | undefined> | undefined;
        let embeddingModel: string | undefined;

        if (embedder) {
          embeddings = [];
          embeddingModel = env.openaiEmbeddingModel;
          for (const content of chunkContents) {
            const e = await embedder.embed(content);
            embeddings.push(e);
          }
        }

        const index = buildIndexFile({
          personaId: doc.persona_id,
          chunkContents,
          embeddingModel,
          embeddings
        });

        await writeJsonPretty(env.personaIndexPath, index);

        record.status = 'succeeded';
        record.finishedAt = new Date().toISOString();
        record.result = {
          personaId: doc.persona_id,
          docPath: env.personaDocPath,
          indexPath: env.personaIndexPath,
          validationIssueCount: validationIssues.length
        };
      } catch (error) {
        record.status = 'failed';
        record.finishedAt = new Date().toISOString();
        record.error = error instanceof Error ? error.message : 'persona_job_failed';
        logger?.warn({ error, jobId: record.id }, 'persona job failed');
      } finally {
        if (activeJobId === record.id) {
          activeJobId = null;
        }
      }
    })();

    return record;
  }

  return { startCycle, getJob, getActiveJob };
}

