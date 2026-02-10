import 'dotenv/config';

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { loadEnv } from '../../config/env';
import { loadCharacterSchemaFromPath } from '../../persona/loadCharacterSchema';
import { generatePersonaAdversarial } from '../../persona/adversarialPersonaGenerator';
import { createEmbeddingsGateway } from '../../persona/embeddingsGateway';
import { renderPersonaMarkdown } from '../../persona/renderPersona';
import { buildIndexFile, chunkText } from '../../persona/personaIndex';
import { pickDefaultPersonaModels } from '../../persona/personaModelClients';
import type { PersonaModelSpec, PersonaProvider } from '../../persona/personaModelClients';
import { resolveRepoPath } from '../../persona/repoPaths';

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseProvider(value: string | undefined): PersonaProvider | null {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === 'openai' || normalized === 'anthropic') return normalized;
  return null;
}

async function writeJsonPretty(filePath: string, data: unknown): Promise<void> {
  const abs = resolveRepoPath(filePath);
  await mkdir(path.dirname(abs), { recursive: true });
  await writeFile(abs, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

function envModelSpec(prefix: string): Partial<PersonaModelSpec> {
  const provider = parseProvider(process.env[`${prefix}_PROVIDER`]);
  const model = process.env[`${prefix}_MODEL`];
  return {
    ...(provider ? { provider } : {}),
    ...(model ? { model } : {})
  };
}

async function main(): Promise<void> {
  const env = loadEnv();

  const personaId = process.env.PERSONA_ID ?? 'default';
  const iterations = parseNumber(process.env.PERSONA_ITERATIONS, 2);
  const schemaPath = resolveRepoPath(process.env.PERSONA_SCHEMA_PATH ?? 'model/character.json');

  const base = pickDefaultPersonaModels(env);
  const genOverride = envModelSpec('PERSONA_GEN');
  const criticOverride = envModelSpec('PERSONA_CRITIC');

  const generator: PersonaModelSpec = {
    provider: (genOverride.provider ?? base.generator.provider) as PersonaProvider,
    model: genOverride.model ?? base.generator.model
  };

  const critic: PersonaModelSpec = {
    provider: (criticOverride.provider ?? base.critic.provider) as PersonaProvider,
    model: criticOverride.model ?? base.critic.model
  };

  const schema = await loadCharacterSchemaFromPath(schemaPath);

  const { doc, validationIssues } = await generatePersonaAdversarial({
    env,
    schema,
    personaId,
    generator,
    critic,
    iterations
  });

  await writeJsonPretty(env.personaDocPath, doc);

  // Build a retrieval index over the rendered persona.
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

  const issueCount = validationIssues.length;
  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        ok: issueCount === 0,
        personaId: doc.persona_id,
        docPath: env.personaDocPath,
        indexPath: env.personaIndexPath,
        validationIssueCount: issueCount
      },
      null,
      2
    )
  );

  if (issueCount > 0) {
    // eslint-disable-next-line no-console
    console.log('Validation issues (first 20):');
    for (const issue of validationIssues.slice(0, 20)) {
      // eslint-disable-next-line no-console
      console.log(`- ${issue.path}: ${issue.reason}${issue.detail ? ` (${issue.detail})` : ''}`);
    }
    process.exitCode = 2;
  }
}

void main().catch((error) => {
  // eslint-disable-next-line no-console
  console.error(error);
  process.exit(1);
});
