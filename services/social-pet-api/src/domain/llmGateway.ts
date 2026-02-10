import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import type { ModelProvider, SessionState } from '@social-pet/domain';

import type { ApiEnv } from '../config/env';
import { conversationUrgency } from './phase1Engines';

export type ConversationTurn = {
  role: 'user' | 'assistant';
  content: string;
};

export type LLMInput = {
  userMessage: string;
  state: SessionState;
  tone: 'supportive' | 'neutral' | 'harsh';
  history: ConversationTurn[];
  previousProviderResponseId?: string;
  personaContext?: string;
};

export type LLMOutput = {
  text: string;
  provider: ModelProvider;
  model: string;
  latencyMs: number;
  fallback: boolean;
  providerResponseId?: string;
};

export interface LLMGateway {
  generateReply: (input: LLMInput) => Promise<LLMOutput>;
  streamReply: (
    input: LLMInput,
    onToken: (delta: string, accumulated: string) => void,
    options?: { signal?: AbortSignal }
  ) => Promise<LLMOutput>;
}

function timeoutError(ms: number): Error {
  return new Error(`llm timeout after ${ms}ms`);
}

function makeAbortController(
  timeoutMs: number,
  externalSignal?: AbortSignal
): { controller: AbortController; cleanup: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(timeoutError(timeoutMs)), timeoutMs);
  let detachExternalAbort: (() => void) | null = null;

  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort(externalSignal.reason ?? new Error('request_aborted'));
    } else {
      const onExternalAbort = () => {
        controller.abort(externalSignal.reason ?? new Error('request_aborted'));
      };

      externalSignal.addEventListener('abort', onExternalAbort, { once: true });
      detachExternalAbort = () => externalSignal.removeEventListener('abort', onExternalAbort);
    }
  }

  return {
    controller,
    cleanup: () => {
      clearTimeout(timer);
      detachExternalAbort?.();
    }
  };
}

function isAbortError(error: unknown): boolean {
  if (error instanceof Error && error.name === 'AbortError') return true;
  if (error instanceof Error && /abort|aborted|canceled|cancelled|interrupted/i.test(error.message)) {
    return true;
  }
  return false;
}

function fallbackText(input: LLMInput): string {
  if (input.tone === 'supportive') return 'I feel seen. Thank you for being warm with me.';
  if (input.tone === 'harsh') return 'That stung. Can we reset and speak more gently?';
  return 'I am here. Tell me more so I can understand you better.';
}

function systemPrompt(state: SessionState, personaContext?: string): string {
  const lines = [
    'You are an adult companion character in a game.',
    'Primary goal: you and the user gradually get to know each other (mutual understanding).',
    'Keep replies short (1-2 sentences), warm, and appropriate for the current adult life phase.',
    'Ask at most one gentle, open-ended question per reply when it fits.',
    'Occasionally volunteer one small detail about yourself when it fits (do not info-dump).',
    'Never mention internal mechanics, scores, hidden rules, or model providers.',
    ...(personaContext ? ['---', personaContext.trim(), '---'] : []),
    `Current life phase: ${state.stage.mode}`,
    `Current narrative act: ${state.narrative.act}`,
    `Current trust: ${state.bond.trust.toFixed(2)}`
  ];

  // Wind-down hints as the creature approaches its daily interaction limit
  const urgency = conversationUrgency(state.timeline);
  if (urgency === 'winding_down') {
    lines.push(
      'You are starting to feel like you need to go soon. Begin winding down the conversation naturally over the next exchange or two.'
    );
  } else if (urgency === 'final') {
    lines.push(
      'You need to leave now. In this response, say a warm, natural goodbye that fits your personality. Do not ask any questions.'
    );
  } else if (urgency === 'overtime') {
    lines.push(
      'You have already said goodbye. Give a brief, final farewell and end the conversation.'
    );
  }

  return lines.join('\n');
}

function buildOpenAIInput(input: LLMInput): Array<{ role: 'system' | 'user' | 'assistant'; content: string }> {
  return [
    { role: 'system', content: systemPrompt(input.state, input.personaContext) },
    ...input.history.map((turn) => ({ role: turn.role, content: turn.content })),
    { role: 'user', content: input.userMessage }
  ];
}

function buildAnthropicMessages(input: LLMInput): Array<{ role: 'user' | 'assistant'; content: string }> {
  return [
    ...input.history.map((turn) => ({ role: turn.role, content: turn.content })),
    { role: 'user', content: input.userMessage }
  ];
}

function extractAnthropicText(content: Anthropic.Message['content']): string {
  return content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
    .trim();
}

function openAIRequestBody(env: ApiEnv, input: LLMInput): {
  model: string;
  input: string | Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  max_output_tokens: number;
  temperature: number;
  store: false;
  previous_response_id?: string;
  instructions?: string;
} {
  if (input.previousProviderResponseId) {
    return {
      model: env.openaiModel,
      input: input.userMessage,
      previous_response_id: input.previousProviderResponseId,
      instructions: systemPrompt(input.state, input.personaContext),
      max_output_tokens: 120,
      temperature: 0.7,
      store: false
    };
  }

  return {
    model: env.openaiModel,
    input: buildOpenAIInput(input),
    max_output_tokens: 120,
    temperature: 0.7,
    store: false
  };
}

export function createLLMGateway(env: ApiEnv, logger?: { warn: (data: unknown, msg?: string) => void }): LLMGateway {
  const openai = env.openaiApiKey ? new OpenAI({ apiKey: env.openaiApiKey }) : null;
  const anthropic = env.anthropicApiKey ? new Anthropic({ apiKey: env.anthropicApiKey }) : null;

  async function callOpenAINonStream(input: LLMInput): Promise<LLMOutput> {
    const startedAt = Date.now();

    if (!openai) throw new Error('openai_api_key_missing');

    const { controller, cleanup } = makeAbortController(env.llmTimeoutMs);

    try {
      const response = await openai.responses.create(openAIRequestBody(env, input), {
        signal: controller.signal
      });

      const text = response.output_text?.trim();
      if (!text) throw new Error('openai_empty_output');

      return {
        text,
        provider: 'openai',
        model: env.openaiModel,
        latencyMs: Date.now() - startedAt,
        fallback: false,
        providerResponseId: response.id
      };
    } finally {
      cleanup();
    }
  }

  async function callOpenAIStream(
    input: LLMInput,
    onToken: (delta: string, accumulated: string) => void,
    signal?: AbortSignal
  ): Promise<LLMOutput> {
    const startedAt = Date.now();

    if (!openai) throw new Error('openai_api_key_missing');

    const { controller, cleanup } = makeAbortController(env.llmTimeoutMs, signal);

    try {
      const stream = openai.responses.stream(openAIRequestBody(env, input), {
        signal: controller.signal
      });

      let text = '';
      stream.on('response.output_text.delta', (event) => {
        if (controller.signal.aborted) return;
        text += event.delta;
        onToken(event.delta, text);
      });

      const finalResponse = await stream.finalResponse();
      const finalText = (finalResponse.output_text ?? text).trim();
      if (!finalText) throw new Error('openai_empty_stream_output');

      return {
        text: finalText,
        provider: 'openai',
        model: env.openaiModel,
        latencyMs: Date.now() - startedAt,
        fallback: false,
        providerResponseId: finalResponse.id
      };
    } finally {
      cleanup();
    }
  }

  async function callAnthropicNonStream(input: LLMInput): Promise<LLMOutput> {
    const startedAt = Date.now();

    if (!anthropic) throw new Error('anthropic_api_key_missing');

    const { controller, cleanup } = makeAbortController(env.llmTimeoutMs);

    try {
      const response = await anthropic.messages.create(
        {
          model: env.anthropicModel,
          max_tokens: 120,
          temperature: 0.7,
          system: systemPrompt(input.state, input.personaContext),
          messages: buildAnthropicMessages(input)
        },
        { signal: controller.signal }
      );

      const text = extractAnthropicText(response.content);
      if (!text) throw new Error('anthropic_empty_output');

      return {
        text,
        provider: 'anthropic',
        model: env.anthropicModel,
        latencyMs: Date.now() - startedAt,
        fallback: false,
        providerResponseId: response.id
      };
    } finally {
      cleanup();
    }
  }

  async function callAnthropicStream(
    input: LLMInput,
    onToken: (delta: string, accumulated: string) => void,
    signal?: AbortSignal
  ): Promise<LLMOutput> {
    const startedAt = Date.now();

    if (!anthropic) throw new Error('anthropic_api_key_missing');

    const { controller, cleanup } = makeAbortController(env.llmTimeoutMs, signal);

    try {
      const stream = anthropic.messages.stream(
        {
          model: env.anthropicModel,
          max_tokens: 120,
          temperature: 0.7,
          system: systemPrompt(input.state, input.personaContext),
          messages: buildAnthropicMessages(input)
        },
        { signal: controller.signal }
      );

      let text = '';
      stream.on('text', (delta, snapshot) => {
        if (controller.signal.aborted) return;
        text = snapshot;
        onToken(delta, text);
      });

      const finalText = (await stream.finalText()).trim();
      if (!finalText) throw new Error('anthropic_empty_stream_output');

      return {
        text: finalText,
        provider: 'anthropic',
        model: env.anthropicModel,
        latencyMs: Date.now() - startedAt,
        fallback: false,
        providerResponseId: undefined
      };
    } finally {
      cleanup();
    }
  }

  function heuristicResult(input: LLMInput): LLMOutput {
    return {
      text: fallbackText(input),
      provider: 'heuristic',
      model: 'heuristic-fastpath-v1',
      latencyMs: 0,
      fallback: true
    };
  }

  return {
    async generateReply(input: LLMInput): Promise<LLMOutput> {
      try {
        if (env.llmProvider === 'openai') return await callOpenAINonStream(input);
        if (env.llmProvider === 'anthropic') return await callAnthropicNonStream(input);
      } catch (error) {
        logger?.warn({ error, provider: env.llmProvider }, 'provider call failed; using fallback');
      }

      return heuristicResult(input);
    },

    async streamReply(
      input: LLMInput,
      onToken: (delta: string, accumulated: string) => void,
      options?: { signal?: AbortSignal }
    ): Promise<LLMOutput> {
      if (options?.signal?.aborted) {
        throw (options.signal.reason ?? new Error('request_aborted'));
      }

      try {
        if (env.llmProvider === 'openai') return await callOpenAIStream(input, onToken, options?.signal);
        if (env.llmProvider === 'anthropic') {
          return await callAnthropicStream(input, onToken, options?.signal);
        }
      } catch (error) {
        if (isAbortError(error)) throw error;
        logger?.warn({ error, provider: env.llmProvider }, 'provider stream failed; using fallback');
      }

      const fallback = heuristicResult(input);
      onToken(fallback.text, fallback.text);
      return fallback;
    }
  };
}
