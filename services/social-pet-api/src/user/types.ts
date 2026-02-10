export type UserModelDoc = {
  schema_name: string;
  schema_version: string;
  user_id: string;
  updated_at: string;
  // Same top-level category keys as character.json; values may be null when unknown.
  user_character: Record<string, Record<string, unknown> | null>;
  notes?: {
    model?: { provider: 'openai'; model: string };
    updated_from_session_id?: string;
    updated_from_event_id?: string;
  };
};

