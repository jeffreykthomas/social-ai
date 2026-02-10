import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { ApiEnv } from '../../config/env';
import type { GameService } from '../../domain/gameService';

// ── Signed URL ──────────────────────────────────────────────────────────────

// Track the active ConvAI session so the LLM proxy knows which game session to use.
// ElevenLabs doesn't reliably forward user_id/extra_body to the LLM request,
// so we store it server-side when the signed URL is requested.
let activeConvaiSessionId: string | null = null;

export function registerConvaiRoutes(
  app: FastifyInstance,
  gameService: GameService,
  env: ApiEnv
): void {
  /**
   * GET /voice/convai/signed-url
   * Returns a signed WebSocket URL so the client can connect to ElevenLabs
   * without exposing the API key.
   */
  app.get('/voice/convai/signed-url', async (request, reply) => {
    if (!env.elevenlabsApiKey || !env.elevenlabsAgentId) {
      return reply.status(503).send({ error: 'elevenlabs not configured' });
    }

    const res = await fetch(
      `https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id=${env.elevenlabsAgentId}`,
      { headers: { 'xi-api-key': env.elevenlabsApiKey } }
    );

    if (!res.ok) {
      const body = await res.text().catch(() => 'unknown');
      app.log.warn({ status: res.status, body }, 'elevenlabs signed-url failed');
      return reply.status(502).send({ error: 'elevenlabs_signed_url_failed' });
    }

    // Store the active session for the LLM proxy to use
    const query = request.query as { gender?: string; sessionId?: string };
    if (query.sessionId) {
      activeConvaiSessionId = query.sessionId;
    }

    // Pick voice based on persona gender
    const gender = query.gender?.toLowerCase() ?? '';
    const isFemale = gender.includes('woman') || gender.includes('female') || gender.includes('she');
    const pool = isFemale ? env.elevenlabsVoiceIdsFemale : env.elevenlabsVoiceIdsMale;
    const fallbackPool = pool.length > 0 ? pool : [...env.elevenlabsVoiceIdsFemale, ...env.elevenlabsVoiceIdsMale];
    const voiceId = fallbackPool.length > 0
      ? fallbackPool[Math.floor(Math.random() * fallbackPool.length)]
      : undefined;

    const data = (await res.json()) as { signed_url: string };
    return reply.send({
      signedUrl: data.signed_url,
      voiceId
    });
  });

  // ── OpenAI-compatible LLM proxy (called by ElevenLabs) ──────────────────

  const chatCompletionSchema = z.object({
    messages: z.array(
      z.object({
        role: z.enum(['system', 'user', 'assistant']),
        content: z.string()
      })
    ),
    model: z.string().optional(),
    temperature: z.number().optional(),
    max_tokens: z.number().optional(),
    stream: z.boolean().optional(),
    // ElevenLabs passes user_id from conversation init
    user_id: z.string().optional(),
    // Also accept elevenlabs_extra_body as fallback
    elevenlabs_extra_body: z.record(z.unknown()).optional()
  });

  /**
   * POST /v1/chat/completions
   * OpenAI-compatible endpoint. ElevenLabs sends conversation messages here;
   * we extract the latest user message and process it through the full game
   * pipeline (Claude, needs, bond, progression, etc.), returning a streaming
   * SSE response in OpenAI format.
   */
  // ElevenLabs appends /chat/completions to the custom LLM base URL.
  // Register both paths so it works whether the agent URL includes /v1 or not.
  const chatHandler = async (request: import('fastify').FastifyRequest, reply: import('fastify').FastifyReply) => {
    const body = chatCompletionSchema.parse(request.body);

    // Session ID: try user_id, extra_body, then fall back to the server-side stored session
    const sessionId =
      body.user_id ||
      (body.elevenlabs_extra_body as { sessionId?: string } | undefined)?.sessionId ||
      activeConvaiSessionId;
    if (!sessionId) {
      return reply.status(400).send({ error: 'no active session — open the app first' });
    }

    // Extract the latest user message from the conversation
    const lastUserMsg = [...body.messages].reverse().find((m) => m.role === 'user');
    if (!lastUserMsg) {
      return reply.status(400).send({ error: 'no user message found' });
    }

    const completionId = `chatcmpl-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);
    const modelName = 'social-pet-claude';

    // Stream SSE response
    reply.raw.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    });

    try {
      const result = await gameService.streamInteraction(
        sessionId,
        lastUserMsg.content,
        (delta) => {
          const chunk = {
            id: completionId,
            object: 'chat.completion.chunk',
            created,
            model: modelName,
            choices: [{ index: 0, delta: { content: delta }, finish_reason: null }]
          };
          reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
        }
      );

      if (!result) {
        const errChunk = {
          id: completionId,
          object: 'chat.completion.chunk',
          created,
          model: modelName,
          choices: [
            { index: 0, delta: { content: 'I seem to have lost my train of thought.' }, finish_reason: 'stop' }
          ]
        };
        reply.raw.write(`data: ${JSON.stringify(errChunk)}\n\n`);
      } else {
        // Final stop chunk
        const stopChunk = {
          id: completionId,
          object: 'chat.completion.chunk',
          created,
          model: modelName,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
        };
        reply.raw.write(`data: ${JSON.stringify(stopChunk)}\n\n`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : '';
      const isWindowExhausted =
        message === 'session_window_exhausted' || message === 'session_ended' || message === 'timeline_complete';

      const fallbackText = isWindowExhausted
        ? "I've really got to go now. It was lovely talking with you — let's pick this up again next time!"
        : 'Something went wrong on my end.';

      if (!isWindowExhausted) {
        app.log.error({ error }, 'convai llm proxy error');
      }

      const errChunk = {
        id: completionId,
        object: 'chat.completion.chunk',
        created,
        model: modelName,
        choices: [{ index: 0, delta: { content: fallbackText }, finish_reason: 'stop' }]
      };
      reply.raw.write(`data: ${JSON.stringify(errChunk)}\n\n`);
    }

    reply.raw.write('data: [DONE]\n\n');
    reply.raw.end();
  };

  app.post('/v1/chat/completions', chatHandler);
  app.post('/chat/completions', chatHandler);
}
