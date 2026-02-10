import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';

import type { ApiEnv } from '../config/env';

export type PersonaProvider = 'openai' | 'anthropic';

export type PersonaModelSpec = {
  provider: PersonaProvider;
  model: string;
};

export type PersonaChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

export type PersonaTextFormat =
  | { type: 'json_object' }
  | { type: 'json_schema'; name: string; schema: unknown; strict: true };

export type PersonaChatClient = {
  provider: PersonaProvider;
  model: string;
  generateText: (
    messages: PersonaChatMessage[],
    opts?: { maxTokens?: number; temperature?: number; format?: PersonaTextFormat }
  ) => Promise<string>;
};

function extractAnthropicText(content: Anthropic.Message['content']): string {
  return content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
    .trim();
}

export function createPersonaChatClient(env: ApiEnv, spec: PersonaModelSpec): PersonaChatClient {
  if (spec.provider === 'openai') {
    if (!env.openaiApiKey) throw new Error('openai_api_key_missing');
    const openai = new OpenAI({ apiKey: env.openaiApiKey });

    return {
      provider: 'openai',
      model: spec.model,
      async generateText(messages, opts) {
        const maxTokens = opts?.maxTokens ?? 3500;
        const temperature = opts?.temperature ?? 0.7;
        const format: PersonaTextFormat = opts?.format ?? { type: 'json_object' };
        const resp = await openai.responses.create({
          model: spec.model,
          input: messages.map((m) => ({ role: m.role, content: m.content })),
          text: { format: format as unknown as OpenAI.Responses.ResponseFormatTextConfig, verbosity: 'low' },
          max_output_tokens: maxTokens,
          temperature,
          store: false
        });
        const text = resp.output_text?.trim();
        if (!text) throw new Error('openai_empty_output');
        return text;
      }
    };
  }

  if (spec.provider === 'anthropic') {
    if (!env.anthropicApiKey) throw new Error('anthropic_api_key_missing');
    const anthropic = new Anthropic({ apiKey: env.anthropicApiKey });

    return {
      provider: 'anthropic',
      model: spec.model,
      async generateText(messages, opts) {
        const maxTokens = opts?.maxTokens ?? 3500;
        const temperature = opts?.temperature ?? 0.7;
        // Anthropic SDK doesn't support Responses API response_format equivalents here; rely on prompts.
        void opts?.format;

        const system = messages.find((m) => m.role === 'system')?.content ?? '';
        const rest = messages.filter((m) => m.role !== 'system');
        const anthropicMessages: Anthropic.MessageParam[] = rest.map((m): Anthropic.MessageParam => ({
          role: m.role === 'assistant' ? 'assistant' : 'user',
          content: m.content
        }));

        const resp = await anthropic.messages.create({
          model: spec.model,
          system,
          messages: anthropicMessages,
          max_tokens: maxTokens,
          temperature
        });

        const text = extractAnthropicText(resp.content);
        if (!text) throw new Error('anthropic_empty_output');
        return text;
      }
    };
  }

  throw new Error(`unknown_provider:${String((spec as { provider: unknown }).provider)}`);
}

export function pickDefaultPersonaModels(env: ApiEnv): { generator: PersonaModelSpec; critic: PersonaModelSpec } {
  // Prefer a genuinely adversarial pairing when both keys are present.
  if (env.openaiApiKey && env.anthropicApiKey) {
    return {
      generator: { provider: 'openai', model: 'gpt-5.2' },
      critic: { provider: 'anthropic', model: 'claude-opus-4-5' }
    };
  }

  if (env.openaiApiKey) {
    return {
      generator: { provider: 'openai', model: 'gpt-5.2' },
      critic: { provider: 'openai', model: 'gpt-5.2' }
    };
  }

  if (env.anthropicApiKey) {
    return {
      generator: { provider: 'anthropic', model: env.anthropicModel },
      critic: { provider: 'anthropic', model: 'claude-opus-4-5' }
    };
  }

  // Script will fail before calling if no keys are available; keep a stable default.
  return {
    generator: { provider: 'openai', model: 'gpt-5.2' },
    critic: { provider: 'openai', model: 'gpt-5.2' }
  };
}
