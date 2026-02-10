import type { CharacterFieldSchema, CharacterSchema, GeneratedPersonaDoc } from './types';
import { parseJsonLoose } from './json';
import { schemaForPrompt } from './loadCharacterSchema';
import type { PersonaModelSpec } from './personaModelClients';
import { createPersonaChatClient } from './personaModelClients';
import type { PersonaTextFormat } from './personaModelClients';
import { characterSchemaToPersonaJsonSchema } from './characterToJsonSchema';
import { validatePersonaAgainstSchema } from './validatePersona';
import type { ApiEnv } from '../config/env';

type CritiquePayload = {
  critique: string;
  must_fix: string[];
  should_fix: string[];
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function collectNullPaths(value: unknown, basePath = ''): string[] {
  if (!isPlainObject(value)) return [];
  const paths: string[] = [];
  for (const [k, v] of Object.entries(value)) {
    const path = basePath ? `${basePath}.${k}` : k;
    if (v === null) paths.push(path);
    if (isPlainObject(v)) paths.push(...collectNullPaths(v, path));
  }
  return paths;
}

function formatList(items: string[], limit = 60): string {
  if (items.length === 0) return '[]';
  const head = items.slice(0, limit).map((p) => `- ${p}`).join('\n');
  const tail = items.length > limit ? `\n- ...and ${items.length - limit} more` : '';
  return `${head}${tail}`;
}

function defaultValueForField(field: CharacterFieldSchema): unknown {
  switch (field.type) {
    case 'string':
      return '<<fill>>';
    case 'string_optional':
      return null;
    case 'number':
      return 0;
    case 'number_optional':
      return null;
    case 'enum':
      return field.options?.[0] ?? '<<fill>>';
    case 'scale_0_10':
      return 5;
    case 'list_string':
      return ['<<fill>>'];
    case 'object': {
      if (!field.schema) return {};
      const obj: Record<string, unknown> = {};
      for (const [k, t] of Object.entries(field.schema)) {
        obj[k] = t === 'scale_0_10' ? 5 : t.endsWith('_optional') ? null : '<<fill>>';
      }
      return obj;
    }
    case 'list_object': {
      if (!field.schema) return [];
      const entry: Record<string, unknown> = {};
      for (const [k, t] of Object.entries(field.schema)) {
        entry[k] = t === 'scale_0_10' ? 5 : t.endsWith('_optional') ? null : '<<fill>>';
      }
      return [entry];
    }
    default:
      return null;
  }
}

export function buildPersonaTemplate(schema: CharacterSchema): Record<string, Record<string, unknown>> {
  const persona: Record<string, Record<string, unknown>> = {};
  for (const cat of schema.top_level_categories) {
    const obj: Record<string, unknown> = {};
    for (const field of cat.fields) {
      obj[field.key] = defaultValueForField(field);
    }
    persona[cat.key] = obj;
  }
  return persona;
}

function personaSystemPrompt(): string {
  return [
    'You are an expert character writer building a rich, distinct social AI companion character.',
    'Output STRICT JSON only (no markdown, no comments).',
    'Follow the schema and template keys exactly. Do not add or remove keys.',
    'Replace placeholder tokens like "<<fill>>" with specific, concrete content.',
    'Prefer contradictions/tensions over one-note traits (e.g., wants intimacy + fears dependence).',
    'Avoid stereotypes; keep demographics tasteful and optional.',
    'Realism constraints: this is an average human-like adult with a life outside the user.',
    'They have mundane routines, obligations, and hobbies that compete with socializing.',
    'Do NOT make their identity or goals revolve around "talking to the user" or being constantly available.',
    'Keep social appetite plausible; avoid extreme attachment/validation hunger unless balanced by constraints and costs.',
    'Minimize nulls: for *_optional fields, provide plausible specific values unless it is truly unknown/private.',
    'Keep content concise: each string should be 1-2 sentences max; list items should be short phrases; lists should be 3-7 items unless the field implies otherwise.',
    'Write in English.'
  ].join('\n');
}

function critiqueSystemPrompt(): string {
  return [
    'You are a strict character sheet critic and schema validator.',
    'Your job: find missing/invalid fields, shallow content, contradictions that reduce believability, and schema noncompliance.',
    'Output STRICT JSON only with keys: critique, must_fix, should_fix.',
    'must_fix: items that block acceptance (missing/invalid schema fields, placeholders, broken structure).',
    'should_fix: improvements for depth, tensions, voice, and actionable behavioral predictions.',
    'Realism checks: flag "companion-bot vibes" where they seem primarily motivated to talk to the user.',
    'Require concrete mundane life details (routines, obligations, hobbies, tastes) that imply they are not always available.',
    'Specifically ensure these are concrete and non-generic: habits_routines_environment.daily_routines, aesthetics_taste_preferences.hobbies_play, aesthetics_taste_preferences.music_taste, goals_plans_obligations.obligations.'
  ].join('\n');
}

function summarySystemPrompt(): string {
  return [
    'You are a character editor producing compact summaries for a game runtime.',
    'Output STRICT JSON only with keys: quick_summary, prompt_summary.',
    'quick_summary: 200-350 words, readable for humans.',
    'prompt_summary: 600-1200 characters, optimized as a system-prompt addendum (voice, motivations, boundaries, social style).',
    'Do not mention schemas, keys, or JSON; do not include bullet nesting deeper than 1 level.'
  ].join('\n');
}

function renderIssuesForPrompt(issues: Array<{ path: string; reason: string; detail?: string }>): string {
  if (issues.length === 0) return '[]';
  const lines = issues.slice(0, 80).map((i) => `- ${i.path}: ${i.reason}${i.detail ? ` (${i.detail})` : ''}`);
  const tail = issues.length > 80 ? `\n- ...and ${issues.length - 80} more` : '';
  return `${lines.join('\n')}${tail}`;
}

async function parseOrRepairJson<T>(params: {
  client: { generateText: (messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>, opts?: { maxTokens?: number; temperature?: number }) => Promise<string> };
  rawText: string;
  repairContext: { schema: unknown; template: unknown; note: string };
}): Promise<T> {
  try {
    return parseJsonLoose<T>(params.rawText);
  } catch (error) {
    const schemaStr = JSON.stringify(params.repairContext.schema);
    const templateStr = JSON.stringify(params.repairContext.template);
    const includeSchema = schemaStr.length <= 6000;
    const includeTemplate = templateStr.length <= 6000;

    const repaired = await params.client.generateText(
      [
        {
          role: 'system',
          content: [
            'You are a JSON repair tool.',
            'Output STRICT JSON only (no markdown, no comments).',
            'You MUST return a single valid JSON object.',
            'The output must match the TEMPLATE keys exactly: do not add/remove keys.',
            'If the prior output was truncated, keep values concise but keep every key present.'
          ].join('\n')
        },
        {
          role: 'user',
          content: [
            params.repairContext.note,
            '',
            ...(includeSchema ? ['SCHEMA (structural requirements):', schemaStr, ''] : []),
            ...(includeTemplate ? ['TEMPLATE (keys must match exactly):', templateStr, ''] : []),
            'BROKEN_JSON_TO_REPAIR:',
            params.rawText
          ].join('\n')
        }
      ],
      { maxTokens: 4500, temperature: 0.2 }
    );

    return parseJsonLoose<T>(repaired);
  }
}

export async function generatePersonaAdversarial(params: {
  env: ApiEnv;
  schema: CharacterSchema;
  personaId: string;
  generator: PersonaModelSpec;
  critic: PersonaModelSpec;
  iterations: number;
}): Promise<{ doc: GeneratedPersonaDoc; validationIssues: ReturnType<typeof validatePersonaAgainstSchema>['issues'] }> {
  const { env, schema, personaId, generator, critic, iterations } = params;

  const generatorClient = createPersonaChatClient(env, generator);
  const criticClient = createPersonaChatClient(env, critic);

  const promptSchema = schemaForPrompt(schema);
  const template = buildPersonaTemplate(schema);
  const personaJsonSchema = characterSchemaToPersonaJsonSchema(schema);

  const personaFormat: PersonaTextFormat | undefined =
    generatorClient.provider === 'openai'
      ? {
          type: 'json_schema',
          name: 'persona',
          schema: personaJsonSchema,
          strict: true
        }
      : undefined;

  const initialText = await generatorClient.generateText(
    [
      { role: 'system', content: personaSystemPrompt() },
      {
        role: 'user',
        content: [
          'Create a complete persona object that satisfies this schema.',
          'Return the persona JSON object only (not wrapped).',
          '',
          'SCHEMA (structural requirements):',
          JSON.stringify(promptSchema)
        ].join('\n')
      }
    ],
    { maxTokens: 6500, temperature: 0.75, format: personaFormat }
  );

  let persona = await parseOrRepairJson<Record<string, Record<string, unknown>>>({
    client: generatorClient,
    rawText: initialText,
    repairContext: {
      schema: promptSchema,
      template,
      note: 'Repair the persona JSON output to match the template exactly.'
    }
  });

  let validation = validatePersonaAgainstSchema(schema, persona);
  let lastCritique: CritiquePayload | null = null;

  // Always run at least one critic pass; otherwise semantic issues slip through once structure validates.
  const criticPasses = Math.max(1, Math.max(0, iterations) + 1);
  for (let i = 0; i < criticPasses; i += 1) {
    const nullPaths = collectNullPaths(persona);
    const nullPathsOutsideDemographics = nullPaths.filter((p) => !p.startsWith('demographics_social_position.'));
    const tooManyNulls = nullPathsOutsideDemographics.length > 8;

    const critiqueText = await criticClient.generateText(
      [
        { role: 'system', content: critiqueSystemPrompt() },
        {
          role: 'user',
          content: [
            'SCHEMA:',
            JSON.stringify(promptSchema),
            '',
            'CURRENT_PERSONA:',
            JSON.stringify(persona),
            '',
            'VALIDATION_ISSUES:',
            renderIssuesForPrompt(validation.issues),
            '',
            'NULL_FIELDS (should be minimized; fill unless truly unknown/private):',
            formatList(nullPathsOutsideDemographics),
            '',
            'Return critique JSON now.'
          ].join('\n')
        }
      ],
      { maxTokens: 2000, temperature: 0.35 }
    );

    const critique = await parseOrRepairJson<CritiquePayload>({
      client: criticClient,
      rawText: critiqueText,
      repairContext: {
        schema: { critique: 'string', must_fix: 'string[]', should_fix: 'string[]' },
        template: { critique: '', must_fix: [], should_fix: [] },
        note: 'Repair this critique JSON to be valid and match the required keys.'
      }
    });
    lastCritique = critique;

    const needsRevision =
      !validation.ok ||
      (Array.isArray(critique.must_fix) && critique.must_fix.length > 0) ||
      tooManyNulls;
    if (!needsRevision) break;

    const revisedText = await generatorClient.generateText(
      [
        { role: 'system', content: personaSystemPrompt() },
        {
          role: 'user',
          content: [
            'Revise CURRENT_PERSONA to address the CRITIQUE.',
            'Return JSON matching the template object ONLY (not wrapped).',
            'Do not add/remove keys; do not leave "<<fill>>" placeholders.',
            'Fill null optional fields with plausible specifics unless truly unknown/private (avoid companion-bot vibes).',
            '',
            'CURRENT_PERSONA:',
            JSON.stringify(persona),
            '',
            'CRITIQUE:',
            JSON.stringify(critique),
            '',
            'NULL_FIELDS_TO_FILL:',
            formatList(nullPathsOutsideDemographics)
          ].join('\n')
        }
      ],
      { maxTokens: 6500, temperature: 0.55, format: personaFormat }
    );

    persona = await parseOrRepairJson<Record<string, Record<string, unknown>>>({
      client: generatorClient,
      rawText: revisedText,
      repairContext: {
        schema: promptSchema,
        template,
        note: 'Repair the revised persona JSON to match the template exactly.'
      }
    });
    validation = validatePersonaAgainstSchema(schema, persona);
  }

  const summaryText = await generatorClient.generateText(
    [
      { role: 'system', content: summarySystemPrompt() },
      {
        role: 'user',
        content: [
          'Produce summaries for this persona.',
          'PERSONA:',
          JSON.stringify(persona)
        ].join('\n')
      }
    ],
    {
      maxTokens: 1400,
      temperature: 0.5,
      format:
        generatorClient.provider === 'openai'
          ? { type: 'json_schema', name: 'persona_summary', strict: true, schema: {
              type: 'object',
              properties: {
                quick_summary: { type: 'string' },
                prompt_summary: { type: 'string' }
              },
              required: ['quick_summary', 'prompt_summary'],
              additionalProperties: false
            } }
          : undefined
    }
  );

  const summary = await parseOrRepairJson<{ quick_summary: string; prompt_summary: string }>({
    client: generatorClient,
    rawText: summaryText,
    repairContext: {
      schema: { quick_summary: 'string', prompt_summary: 'string' },
      template: { quick_summary: '', prompt_summary: '' },
      note: 'Repair the summary JSON to be valid and match the required keys.'
    }
  });

  const doc: GeneratedPersonaDoc = {
    schema_name: schema.schema_name,
    schema_version: schema.schema_version,
    persona_id: personaId,
    generated_at: new Date().toISOString(),
    persona,
    quick_summary: String(summary.quick_summary ?? '').trim(),
    prompt_summary: String(summary.prompt_summary ?? '').trim(),
    notes: {
      generator: { provider: generatorClient.provider, model: generatorClient.model },
      critic: { provider: criticClient.provider, model: criticClient.model },
      iterations: criticPasses,
      validation_issue_count: validation.issues.length
    }
  };

  return { doc, validationIssues: validation.issues };
}
