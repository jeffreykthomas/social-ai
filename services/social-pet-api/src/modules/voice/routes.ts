import type { FastifyInstance } from 'fastify';
import { z } from 'zod';

import type { VoiceService } from '../../domain/voiceService';

const transcribeSchema = z.object({
  audio: z.string().min(1),
  mimeType: z.string().default('audio/webm')
});

const synthesizeSchema = z.object({
  text: z.string().min(1)
});

export function registerVoiceRoutes(app: FastifyInstance, voiceService: VoiceService): void {
  // STT: accept base64-encoded audio, return transcribed text
  app.post(
    '/voice/transcribe',
    { bodyLimit: 10 * 1024 * 1024 },
    async (request, reply) => {
      try {
        const body = transcribeSchema.parse(request.body);
        const buffer = Buffer.from(body.audio, 'base64');
        const text = await voiceService.transcribe(buffer, body.mimeType);
        return reply.send({ text });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'transcription_failed';

        if (message.includes('api_key_missing')) {
          return reply.status(503).send({ error: message });
        }

        throw error;
      }
    }
  );

  // TTS: accept text, return streaming audio/mpeg
  app.post('/voice/synthesize', async (request, reply) => {
    try {
      const body = synthesizeSchema.parse(request.body);
      const audioStream = await voiceService.synthesize(body.text);
      reply.type('audio/mpeg');
      return reply.send(audioStream);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'synthesis_failed';

      if (message.includes('api_key_missing')) {
        return reply.status(503).send({ error: message });
      }

      throw error;
    }
  });
}
