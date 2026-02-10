import { readFile } from 'node:fs/promises';

import { z } from 'zod';

import type { CharacterSchema } from './types';

const fieldSchema = z.object({
  key: z.string().min(1),
  type: z.string().min(1),
  prompt: z.string().optional(),
  options: z.array(z.string()).optional(),
  schema: z.record(z.string(), z.string()).optional()
});

const categorySchema = z.object({
  key: z.string().min(1),
  label: z.string().min(1),
  description: z.string().optional(),
  fields: z.array(fieldSchema)
});

const characterSchemaSchema = z.object({
  schema_name: z.string().min(1),
  schema_version: z.string().min(1),
  top_level_categories: z.array(categorySchema)
});

export async function loadCharacterSchemaFromPath(path: string): Promise<CharacterSchema> {
  const raw = await readFile(path, 'utf8');
  const parsed = JSON.parse(raw) as unknown;
  return characterSchemaSchema.parse(parsed) as CharacterSchema;
}

export function schemaForPrompt(schema: CharacterSchema): unknown {
  // Strip metadata that doesn't help token budget; keep structural requirements.
  return {
    schema_name: schema.schema_name,
    schema_version: schema.schema_version,
    top_level_categories: schema.top_level_categories.map((cat) => ({
      key: cat.key,
      label: cat.label,
      fields: cat.fields.map((f) => ({
        key: f.key,
        type: f.type,
        options: f.options,
        schema: f.schema
      }))
    }))
  };
}

