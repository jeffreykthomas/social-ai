import { Readable } from 'node:stream';

import OpenAI from 'openai';

import type { ApiEnv } from '../config/env';

export interface VoiceService {
  transcribe(audio: Buffer, mimeType: string): Promise<string>;
  synthesize(text: string, options?: { gender?: string }): Promise<Readable>;
}

export function createVoiceService(
  env: ApiEnv,
  logger?: { warn: (data: unknown, msg?: string) => void }
): VoiceService {
  const openai = env.openaiApiKey ? new OpenAI({ apiKey: env.openaiApiKey }) : null;

  return {
    async transcribe(audio: Buffer, mimeType: string): Promise<string> {
      if (!openai) throw new Error('openai_api_key_missing (required for Whisper STT)');

      const ext = mimeType.includes('webm') ? 'webm' : mimeType.includes('mp4') ? 'mp4' : 'webm';
      const file = new File([new Uint8Array(audio)], `audio.${ext}`, { type: mimeType });

      const result = await openai.audio.transcriptions.create({
        file,
        model: 'whisper-1'
      });

      return result.text;
    },

    async synthesize(text: string, options?: { gender?: string }): Promise<Readable> {
      if (!env.elevenlabsApiKey) throw new Error('elevenlabs_api_key_missing');

      const gender = (options?.gender ?? '').toLowerCase();
      const isFemale = gender.includes('woman') || gender.includes('female') || gender.includes('she');
      const pool = isFemale ? env.elevenlabsVoiceIdsFemale : env.elevenlabsVoiceIdsMale;
      const fallback = pool.length > 0 ? pool : [...env.elevenlabsVoiceIdsFemale, ...env.elevenlabsVoiceIdsMale];
      const voiceId = fallback.length > 0
        ? fallback[Math.floor(Math.random() * fallback.length)]
        : '';

      if (!voiceId) throw new Error('no_voice_ids_configured');

      const response = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`,
        {
          method: 'POST',
          headers: {
            'xi-api-key': env.elevenlabsApiKey,
            'Content-Type': 'application/json',
            Accept: 'audio/mpeg'
          },
          body: JSON.stringify({
            text,
            model_id: env.elevenlabsModelId,
            voice_settings: {
              stability: 0.5,
              similarity_boost: 0.75,
              style: 0.0,
              use_speaker_boost: true
            }
          })
        }
      );

      if (!response.ok) {
        const errorBody = await response.text().catch(() => 'unknown');
        logger?.warn(
          { status: response.status, body: errorBody },
          'elevenlabs tts failed'
        );
        throw new Error(`elevenlabs_tts_failed: ${response.status}`);
      }

      if (!response.body) throw new Error('elevenlabs_empty_response');

      // Convert web ReadableStream to Node.js Readable for Fastify
      return Readable.fromWeb(response.body as Parameters<typeof Readable.fromWeb>[0]);
    }
  };
}
