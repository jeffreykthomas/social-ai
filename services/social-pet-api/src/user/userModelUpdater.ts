import OpenAI from 'openai';
import type { InteractionEvent, SessionState } from '@social-pet/domain';

import type { ApiEnv } from '../config/env';
import { loadCharacterSchemaFromPath, schemaForPrompt } from '../persona/loadCharacterSchema';
import { characterSchemaToUserModelJsonSchema } from '../persona/characterToJsonSchema';
import { resolveRepoPath } from '../persona/repoPaths';
import { loadUserModelDoc, writeUserModelDoc } from './userModelFiles';
import type { UserModelDoc } from './types';

function buildEmptyUserCharacter(schema: { top_level_categories: Array<{ key: string; fields: Array<{ key: string }> }> }): Record<
  string,
  Record<string, unknown> | null
> {
  const obj: Record<string, Record<string, unknown> | null> = {};
  for (const cat of schema.top_level_categories) {
    const fields: Record<string, unknown> = {};
    for (const f of cat.fields) fields[f.key] = null;
    obj[cat.key] = fields;
  }
  return obj;
}

function systemPrompt(): string {
  return [
    'You are a memory editor that maintains a structured model of the USER (player).',
    'You update fields only when supported by evidence from the transcript/facts provided.',
    'If a field is unknown or ambiguous, output null for that field.',
    'Do not invent demographics, trauma, medical conditions, or sensitive attributes without explicit evidence.',
    'Prefer concise values: strings 1-2 sentences, list items short phrases.',
    'Output JSON only.'
  ].join('\n');
}

function buildEvidence(events: InteractionEvent[], state: SessionState): string {
  const facts = state.knowledge?.userFacts ?? [];
  const factLines = facts
    .slice(-40)
    .map((f) => `- (${f.category}) ${f.fact}`)
    .join('\n');

  const recent = events.slice(-20).map((e) => `User: ${e.userMessage}\nAssistant: ${e.responseText}`).join('\n\n');

  return [
    'RECENT_USER_FACTS:',
    factLines.length > 0 ? factLines : '(none)',
    '',
    'RECENT_TRANSCRIPT_EXCERPTS:',
    recent.length > 0 ? recent : '(none)'
  ].join('\n');
}

export function createUserModelUpdater(env: ApiEnv, logger?: { warn: (data: unknown, msg?: string) => void }): {
  updateFromSession: (input: { userId: string; sessionId: string; state: SessionState; events: InteractionEvent[]; lastEvent: InteractionEvent }) => Promise<UserModelDoc>;
} {
  const openai = env.openaiApiKey ? new OpenAI({ apiKey: env.openaiApiKey }) : null;
  let schemaCache: Awaited<ReturnType<typeof loadCharacterSchemaFromPath>> | null = null;

  async function loadSchema() {
    if (schemaCache) return schemaCache;
    schemaCache = await loadCharacterSchemaFromPath(resolveRepoPath('model/character.json'));
    return schemaCache;
  }

  return {
    async updateFromSession(input): Promise<UserModelDoc> {
      if (!openai) throw new Error('openai_api_key_missing');
      if (!env.userModelEnabled) throw new Error('user_model_disabled');

      const schema = await loadSchema();
      const promptSchema = schemaForPrompt(schema);
      const outputSchema = characterSchemaToUserModelJsonSchema(schema);

      const existing = await loadUserModelDoc(env.userModelDir, input.userId);
      const empty = buildEmptyUserCharacter(schema);
      const baseCharacter = existing?.user_character ?? empty;

      const resp = await openai.responses.create({
        model: env.userModelOpenAIModel,
        input: [
          { role: 'system', content: systemPrompt() },
          {
            role: 'user',
            content: [
              'Update USER_CHARACTER based on EVIDENCE. Return the full USER_CHARACTER object (top-level categories).',
              'Do not add or remove keys.',
              '',
              'SCHEMA (structural requirements):',
              JSON.stringify(promptSchema),
              '',
              'CURRENT_USER_CHARACTER:',
              JSON.stringify(baseCharacter),
              '',
              'EVIDENCE:',
              buildEvidence(input.events, input.state)
            ].join('\n')
          }
        ],
        text: {
          format: {
            type: 'json_schema',
            name: 'user_character',
            schema: outputSchema,
            strict: true
          },
          verbosity: 'low'
        },
        max_output_tokens: 4500,
        temperature: 0.2,
        store: false
      });

      const jsonText = resp.output_text?.trim();
      if (!jsonText) throw new Error('openai_empty_output');

      let updatedCharacter: Record<string, Record<string, unknown> | null>;
      try {
        updatedCharacter = JSON.parse(jsonText) as Record<string, Record<string, unknown> | null>;
      } catch (error) {
        logger?.warn({ error }, 'user model parse failed');
        throw new Error('user_model_invalid_json');
      }

      const nextDoc: UserModelDoc = {
        schema_name: schema.schema_name,
        schema_version: schema.schema_version,
        user_id: input.userId,
        updated_at: new Date().toISOString(),
        user_character: updatedCharacter,
        notes: {
          model: { provider: 'openai', model: env.userModelOpenAIModel },
          updated_from_session_id: input.sessionId,
          updated_from_event_id: input.lastEvent.id
        }
      };

      await writeUserModelDoc(env.userModelDir, input.userId, nextDoc);
      return nextDoc;
    }
  };
}

