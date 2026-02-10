export function extractJsonObject(text: string): string {
  const trimmed = text.trim();

  // If it's already JSON, return.
  if (trimmed.startsWith('{') && trimmed.endsWith('}')) return trimmed;
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) return trimmed;

  // Try to extract from ```json ... ```
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced?.[1]) return fenced[1].trim();

  // Fallback: first balanced { ... } span.
  const firstBrace = trimmed.indexOf('{');
  if (firstBrace === -1) return trimmed;

  let depth = 0;
  for (let i = firstBrace; i < trimmed.length; i += 1) {
    const ch = trimmed[i];
    if (ch === '{') depth += 1;
    if (ch === '}') depth -= 1;
    if (depth === 0) return trimmed.slice(firstBrace, i + 1).trim();
  }

  return trimmed.slice(firstBrace).trim();
}

export function parseJsonLoose<T>(text: string): T {
  const extracted = extractJsonObject(text);
  return JSON.parse(extracted) as T;
}

