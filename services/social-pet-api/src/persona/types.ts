export type CharacterFieldType =
  | 'string'
  | 'string_optional'
  | 'number'
  | 'number_optional'
  | 'enum'
  | 'enum_short_mid_long'
  | 'scale_0_10'
  | 'list_string'
  | 'list_string_optional'
  | 'list_object'
  | 'object'
  | 'object_optional';

export type CharacterFieldSchema = {
  key: string;
  type: CharacterFieldType | string;
  prompt?: string;
  options?: string[];
  // For type=object, schema is a map of key -> field type (e.g. scale_0_10)
  // For type=list_object, schema is a map of key -> field type (e.g. string_optional)
  schema?: Record<string, string>;
};

export type CharacterCategorySchema = {
  key: string;
  label: string;
  description?: string;
  fields: CharacterFieldSchema[];
};

export type CharacterSchema = {
  schema_name: string;
  schema_version: string;
  top_level_categories: CharacterCategorySchema[];
};

export type GeneratedPersonaDoc = {
  schema_name: string;
  schema_version: string;
  persona_id: string;
  generated_at: string;
  // Keyed by category key, each containing keyed field values.
  persona: Record<string, Record<string, unknown>>;
  // Human-readable, compact overview suitable for UI/debug.
  quick_summary: string;
  // Model-facing base prompt summary (kept short to avoid prompt bloat).
  prompt_summary: string;
  notes?: {
    generator?: { provider: string; model: string };
    critic?: { provider: string; model: string };
    iterations?: number;
    validation_issue_count?: number;
  };
};

export type PersonaIndexChunk = {
  id: string;
  chunk_index: number;
  content: string;
  // Omitted when embeddings were not generated.
  embedding?: number[];
};

export type PersonaIndexFile = {
  persona_id: string;
  created_at: string;
  embedding_model?: string;
  chunks: PersonaIndexChunk[];
};
