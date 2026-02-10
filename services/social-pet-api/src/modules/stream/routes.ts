import type { FastifyInstance } from 'fastify';
import type { RawData, WebSocket } from 'ws';

import type { GameService } from '../../domain/gameService';

type StreamInMessage =
  | {
      type: 'user_message';
      sessionId: string;
      message: string;
    }
  | {
      type: 'stop_stream';
    }
  | {
      type: 'ping';
    };

type StreamOutMessage =
  | { type: 'ready' }
  | { type: 'pong' }
  | { type: 'stream_started'; at: string }
  | { type: 'stream_stopped'; reason: 'client_stop' | 'socket_closed' | 'new_message' | 'idle' }
  | { type: 'assistant_delta'; delta: string; text: string }
  | {
      type: 'assistant_done';
      payload: {
        state: unknown;
        responseText: string;
        event: unknown;
        meta: {
          totalLatencyMs: number;
          modelLatencyMs: number;
          usedFallback: boolean;
        };
      };
    }
  | { type: 'error'; error: string };

type ActiveRun = {
  id: number;
  controller: AbortController;
};

function decodeRaw(raw: RawData): string {
  if (typeof raw === 'string') return raw;
  if (raw instanceof Buffer) return raw.toString('utf8');
  if (Array.isArray(raw)) return Buffer.concat(raw).toString('utf8');
  if (raw instanceof ArrayBuffer) return Buffer.from(new Uint8Array(raw)).toString('utf8');
  if (ArrayBuffer.isView(raw)) {
    return Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength).toString('utf8');
  }
  return String(raw);
}

function send(socket: WebSocket, message: StreamOutMessage): void {
  if (socket.readyState !== 1) return;
  socket.send(JSON.stringify(message));
}

function isAbortError(error: unknown): boolean {
  if (error instanceof Error && error.name === 'AbortError') return true;
  if (error instanceof Error && /abort|aborted|interrupted|canceled|cancelled/i.test(error.message)) {
    return true;
  }
  return false;
}

function parseMessage(raw: RawData): StreamInMessage | null {
  try {
    const parsed = JSON.parse(decodeRaw(raw)) as Partial<StreamInMessage>;

    if (parsed.type === 'ping') return { type: 'ping' };
    if (parsed.type === 'stop_stream') return { type: 'stop_stream' };

    if (
      parsed.type === 'user_message' &&
      typeof parsed.sessionId === 'string' &&
      parsed.sessionId.length > 0 &&
      typeof parsed.message === 'string' &&
      parsed.message.trim().length > 0
    ) {
      return {
        type: 'user_message',
        sessionId: parsed.sessionId,
        message: parsed.message
      };
    }

    return null;
  } catch {
    return null;
  }
}

export function registerStreamRoutes(app: FastifyInstance, gameService: GameService): void {
  app.get('/interaction/stream/ws', { websocket: true }, (socket) => {
    let activeRun: ActiveRun | null = null;
    let runCounter = 0;

    send(socket, { type: 'ready' });

    socket.on('close', () => {
      activeRun?.controller.abort(new Error('socket_closed'));
      send(socket, { type: 'stream_stopped', reason: 'socket_closed' });
      activeRun = null;
    });

    async function startStream(sessionId: string, userMessage: string): Promise<void> {
      const runId = runCounter + 1;
      runCounter = runId;

      if (activeRun) {
        activeRun.controller.abort(new Error('stream_interrupted_by_new_message'));
        send(socket, { type: 'stream_stopped', reason: 'new_message' });
      }

      const controller = new AbortController();
      activeRun = { id: runId, controller };

      send(socket, { type: 'stream_started', at: new Date().toISOString() });

      try {
        const result = await gameService.streamInteraction(
          sessionId,
          userMessage,
          (delta, text) => {
            if (activeRun?.id !== runId) return;
            send(socket, { type: 'assistant_delta', delta, text });
          },
          { signal: controller.signal }
        );

        if (activeRun?.id !== runId) return;

        if (!result) {
          send(socket, { type: 'error', error: 'session_not_found' });
          return;
        }

        send(socket, {
          type: 'assistant_done',
          payload: result
        });
      } catch (error) {
        if (activeRun?.id !== runId) return;
        if (isAbortError(error)) return;

        const reason = error instanceof Error ? error.message : 'stream_failed';
        send(socket, { type: 'error', error: reason });
      } finally {
        if (activeRun?.id === runId) {
          activeRun = null;
        }
      }
    }

    socket.on('message', (raw: RawData) => {
      const message = parseMessage(raw);

      if (!message) {
        send(socket, { type: 'error', error: 'invalid_message_payload' });
        return;
      }

      if (message.type === 'ping') {
        send(socket, { type: 'pong' });
        return;
      }

      if (message.type === 'stop_stream') {
        if (activeRun) {
          activeRun.controller.abort(new Error('stream_stopped_by_client'));
          send(socket, { type: 'stream_stopped', reason: 'client_stop' });
          return;
        }

        send(socket, { type: 'stream_stopped', reason: 'idle' });
        return;
      }

      void startStream(message.sessionId, message.message);
    });
  });
}
