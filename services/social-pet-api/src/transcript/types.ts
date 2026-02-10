export type TranscriptLine = {
  at: string;
  eventId: string;
  role: 'user' | 'assistant';
  content: string;
};

export type TranscriptIndexChunk = {
  id: string;
  chunk_index: number;
  at: string;
  event_id: string;
  content: string;
  embedding?: number[];
};

export type TranscriptIndexFile = {
  session_id: string;
  created_at: string;
  embedding_model?: string;
  chunks: TranscriptIndexChunk[];
};

