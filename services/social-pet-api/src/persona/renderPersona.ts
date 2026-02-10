import type { CharacterSchema, GeneratedPersonaDoc } from './types';

function safeString(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

export function renderPersonaMarkdown(schema: CharacterSchema, doc: GeneratedPersonaDoc): string {
  const lines: string[] = [];

  lines.push(`# Persona: ${doc.persona_id}`);
  lines.push('');
  lines.push(`Schema: ${doc.schema_name}@${doc.schema_version}`);
  lines.push(`Generated: ${doc.generated_at}`);
  lines.push('');
  lines.push('## Prompt Summary');
  lines.push(doc.prompt_summary.trim());
  lines.push('');
  lines.push('## Quick Summary');
  lines.push(doc.quick_summary.trim());
  lines.push('');

  for (const category of schema.top_level_categories) {
    lines.push(`## ${category.label} (${category.key})`);
    lines.push('');

    const cat = doc.persona[category.key] ?? {};
    for (const field of category.fields) {
      const value = (cat as Record<string, unknown>)[field.key];
      if (Array.isArray(value)) {
        lines.push(`- ${field.key}:`);
        for (const entry of value) {
          lines.push(`  - ${safeString(entry)}`);
        }
        continue;
      }

      if (value && typeof value === 'object') {
        lines.push(`- ${field.key}: ${safeString(value)}`);
        continue;
      }

      lines.push(`- ${field.key}: ${safeString(value)}`);
    }

    lines.push('');
  }

  return lines.join('\n').trimEnd() + '\n';
}

