import type { CharacterFieldSchema, CharacterSchema } from './types';

function requiredKeys(obj: Record<string, unknown>): string[] {
  return Object.keys(obj);
}

function makeNullable(schema: Record<string, unknown>): Record<string, unknown> {
  // Avoid double-wrapping if already nullable.
  const anyOf = schema.anyOf;
  if (Array.isArray(anyOf) && anyOf.some((s) => (s as { type?: unknown }).type === 'null')) {
    return schema;
  }
  return { anyOf: [schema, { type: 'null' }] };
}

function jsonSchemaForType(field: CharacterFieldSchema, opts?: { nullableAll?: boolean }): Record<string, unknown> {
  const nullableAll = Boolean(opts?.nullableAll);

  switch (field.type) {
    case 'string':
      return nullableAll ? makeNullable({ type: 'string', minLength: 1 }) : { type: 'string', minLength: 1 };
    case 'string_optional':
      return { anyOf: [{ type: 'string', minLength: 1 }, { type: 'null' }] };
    case 'number':
      return nullableAll ? makeNullable({ type: 'number' }) : { type: 'number' };
    case 'number_optional':
      return { anyOf: [{ type: 'number' }, { type: 'null' }] };
    case 'scale_0_10':
      return nullableAll ? makeNullable({ type: 'number', minimum: 0, maximum: 10 }) : { type: 'number', minimum: 0, maximum: 10 };
    case 'enum':
      return nullableAll ? makeNullable({ type: 'string', enum: field.options ?? [] }) : { type: 'string', enum: field.options ?? [] };
    case 'enum_short_mid_long':
      return nullableAll ? makeNullable({ type: 'string', enum: ['short', 'mid', 'long'] }) : { type: 'string', enum: ['short', 'mid', 'long'] };
    case 'list_string':
      return nullableAll ? makeNullable({ type: 'array', items: { type: 'string', minLength: 1 }, minItems: 1 }) : { type: 'array', items: { type: 'string', minLength: 1 }, minItems: 1 };
    case 'list_string_optional':
      return { anyOf: [{ type: 'array', items: { type: 'string', minLength: 1 } }, { type: 'null' }] };
    case 'object': {
      const props: Record<string, unknown> = {};
      for (const [k, t] of Object.entries(field.schema ?? {})) {
        props[k] = jsonSchemaForType({ key: k, type: t }, opts);
      }
      const base = { type: 'object', properties: props, required: requiredKeys(props), additionalProperties: false };
      return nullableAll ? makeNullable(base) : base;
    }
    case 'object_optional': {
      const obj = jsonSchemaForType({ ...field, type: 'object' }, opts);
      return { anyOf: [obj, { type: 'null' }] };
    }
    case 'list_object': {
      const itemSchema = jsonSchemaForType({ ...field, type: 'object' }, opts);
      const base = { type: 'array', items: itemSchema, minItems: 1 };
      return nullableAll ? makeNullable(base) : base;
    }
    default:
      // Keep schemas valid for Structured Outputs even if we missed a custom type.
      return nullableAll ? makeNullable({ type: 'string' }) : { type: 'string' };
  }
}

function characterSchemaToJsonSchema(schema: CharacterSchema, opts?: { nullableAll?: boolean }): Record<string, unknown> {
  const categoryProps: Record<string, unknown> = {};

  for (const category of schema.top_level_categories) {
    const fieldProps: Record<string, unknown> = {};
    for (const field of category.fields) {
      fieldProps[field.key] = jsonSchemaForType(field, opts);
    }

    categoryProps[category.key] = {
      type: 'object',
      properties: fieldProps,
      required: requiredKeys(fieldProps),
      additionalProperties: false
    };
  }

  return {
    $schema: 'http://json-schema.org/draft-07/schema#',
    type: 'object',
    properties: categoryProps,
    required: requiredKeys(categoryProps),
    additionalProperties: false
  };
}

export function characterSchemaToPersonaJsonSchema(schema: CharacterSchema): Record<string, unknown> {
  return characterSchemaToJsonSchema(schema, { nullableAll: false });
}

export function characterSchemaToUserModelJsonSchema(schema: CharacterSchema): Record<string, unknown> {
  // Allow null everywhere while still requiring the full key shape.
  return characterSchemaToJsonSchema(schema, { nullableAll: true });
}
