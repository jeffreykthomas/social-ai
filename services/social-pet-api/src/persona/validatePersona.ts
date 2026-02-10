import type { CharacterFieldSchema, CharacterSchema } from './types';

export type PersonaValidationIssue = {
  path: string;
  reason:
    | 'missing'
    | 'wrong_type'
    | 'invalid_enum'
    | 'out_of_range'
    | 'missing_object_key'
    | 'wrong_object_shape';
  detail?: string;
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function typeLabel(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function validateScalarType(field: CharacterFieldSchema, value: unknown, path: string): PersonaValidationIssue[] {
  const issues: PersonaValidationIssue[] = [];

  switch (field.type) {
    case 'string': {
      if (typeof value !== 'string' || value.trim().length === 0) {
        issues.push({ path, reason: 'wrong_type', detail: `expected non-empty string, got ${typeLabel(value)}` });
      }
      return issues;
    }
    case 'string_optional': {
      if (value === null) return issues;
      if (typeof value !== 'string') {
        issues.push({ path, reason: 'wrong_type', detail: `expected string|null, got ${typeLabel(value)}` });
      }
      return issues;
    }
    case 'number': {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        issues.push({ path, reason: 'wrong_type', detail: `expected number, got ${typeLabel(value)}` });
      }
      return issues;
    }
    case 'number_optional': {
      if (value === null) return issues;
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        issues.push({ path, reason: 'wrong_type', detail: `expected number|null, got ${typeLabel(value)}` });
      }
      return issues;
    }
    case 'scale_0_10': {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        issues.push({ path, reason: 'wrong_type', detail: `expected number 0..10, got ${typeLabel(value)}` });
        return issues;
      }
      if (value < 0 || value > 10) {
        issues.push({ path, reason: 'out_of_range', detail: 'expected 0..10' });
      }
      return issues;
    }
    case 'enum': {
      if (typeof value !== 'string') {
        issues.push({ path, reason: 'wrong_type', detail: `expected string enum, got ${typeLabel(value)}` });
        return issues;
      }
      if (field.options && !field.options.includes(value)) {
        issues.push({ path, reason: 'invalid_enum', detail: `expected one of: ${field.options.join(', ')}` });
      }
      return issues;
    }
    case 'enum_short_mid_long': {
      if (typeof value !== 'string') {
        issues.push({ path, reason: 'wrong_type', detail: `expected string enum, got ${typeLabel(value)}` });
        return issues;
      }
      const allowed = ['short', 'mid', 'long'];
      if (!allowed.includes(value)) {
        issues.push({ path, reason: 'invalid_enum', detail: `expected one of: ${allowed.join(', ')}` });
      }
      return issues;
    }
    case 'list_string': {
      if (!Array.isArray(value)) {
        issues.push({ path, reason: 'wrong_type', detail: `expected string[], got ${typeLabel(value)}` });
        return issues;
      }
      for (let i = 0; i < value.length; i += 1) {
        if (typeof value[i] !== 'string') {
          issues.push({ path: `${path}[${i}]`, reason: 'wrong_type', detail: `expected string, got ${typeLabel(value[i])}` });
        }
      }
      return issues;
    }
    case 'list_string_optional': {
      if (value === null) return issues;
      const inner: CharacterFieldSchema = { key: field.key, type: 'list_string' };
      return validateScalarType(inner, value, path);
    }
    case 'object': {
      if (!isPlainObject(value)) {
        issues.push({ path, reason: 'wrong_type', detail: `expected object, got ${typeLabel(value)}` });
        return issues;
      }
      if (!field.schema) return issues;
      for (const [key, childType] of Object.entries(field.schema)) {
        if (!(key in value)) {
          issues.push({ path: `${path}.${key}`, reason: 'missing_object_key' });
          continue;
        }
        const childField: CharacterFieldSchema = { key, type: childType };
        issues.push(...validateScalarType(childField, value[key], `${path}.${key}`));
      }
      return issues;
    }
    case 'object_optional': {
      if (value === null) return issues;
      const inner: CharacterFieldSchema = { ...field, type: 'object' };
      return validateScalarType(inner, value, path);
    }
    case 'list_object': {
      if (!Array.isArray(value)) {
        issues.push({ path, reason: 'wrong_type', detail: `expected object[], got ${typeLabel(value)}` });
        return issues;
      }
      if (!field.schema) return issues;

      for (let i = 0; i < value.length; i += 1) {
        const entry = value[i];
        if (!isPlainObject(entry)) {
          issues.push({ path: `${path}[${i}]`, reason: 'wrong_type', detail: `expected object, got ${typeLabel(entry)}` });
          continue;
        }

        for (const [key, childType] of Object.entries(field.schema)) {
          if (!(key in entry)) {
            issues.push({ path: `${path}[${i}].${key}`, reason: 'missing_object_key' });
            continue;
          }
          const childField: CharacterFieldSchema = { key, type: childType };
          issues.push(...validateScalarType(childField, entry[key], `${path}[${i}].${key}`));
        }
      }
      return issues;
    }
    default: {
      // Unknown schema types are treated as soft errors.
      return issues;
    }
  }
}

export function validatePersonaAgainstSchema(
  schema: CharacterSchema,
  persona: unknown
): { ok: boolean; issues: PersonaValidationIssue[] } {
  const issues: PersonaValidationIssue[] = [];

  if (!isPlainObject(persona)) {
    return { ok: false, issues: [{ path: 'persona', reason: 'wrong_type', detail: `expected object, got ${typeLabel(persona)}` }] };
  }

  for (const category of schema.top_level_categories) {
    const catValue = (persona as Record<string, unknown>)[category.key];
    if (!isPlainObject(catValue)) {
      issues.push({ path: category.key, reason: 'missing', detail: 'missing category object' });
      continue;
    }

    for (const field of category.fields) {
      const fieldPath = `${category.key}.${field.key}`;

      if (!(field.key in catValue)) {
        issues.push({ path: fieldPath, reason: 'missing' });
        continue;
      }

      const value = catValue[field.key];
      issues.push(...validateScalarType(field, value, fieldPath));
    }
  }

  return { ok: issues.length === 0, issues };
}
