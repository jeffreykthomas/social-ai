import 'dotenv/config';
import cors from '@fastify/cors';
import websocket from '@fastify/websocket';
import Fastify from 'fastify';

import { loadEnv } from './config/env';
import { createGameService } from './domain/gameService';
import { createLLMGateway } from './domain/llmGateway';
import { createSessionPersistence } from './domain/persistenceStore';
import { registerAppearanceRoutes } from './modules/appearance/routes';
import { registerHealthRoutes } from './modules/health/routes';
import { registerInteractionRoutes } from './modules/interaction/routes';
import { registerOutreachRoutes } from './modules/outreach/routes';
import { registerPersonaRoutes } from './modules/persona/routes';
import { registerProgressionRoutes } from './modules/progression/routes';
import { registerSessionRoutes } from './modules/session/routes';
import { registerStreamRoutes } from './modules/stream/routes';
import { registerTranscriptRoutes } from './modules/transcript/routes';
import { registerUserRoutes } from './modules/user/routes';
import { registerVoiceRoutes } from './modules/voice/routes';
import { registerConvaiRoutes } from './modules/convai/routes';
import { createVoiceService } from './domain/voiceService';
import { createPersonaRag } from './persona/personaRag';
import { createPersonaJobManager } from './persona/personaJobManager';
import { createTranscriptStore } from './transcript/transcriptStore';
import { createTranscriptRag } from './transcript/transcriptRag';
import { createUserModelJobManager } from './user/userModelJobManager';

async function bootstrap(): Promise<void> {
  const env = loadEnv();
  const app = Fastify({ logger: true });

  await app.register(cors, {
    origin: (origin, cb) => {
      if (!origin) {
        cb(null, true);
        return;
      }

      if (env.corsOrigins.length === 0) {
        cb(null, false);
        return;
      }

      cb(null, env.corsOrigins.includes(origin));
    },
    methods: ['GET', 'POST', 'OPTIONS']
  });
  await app.register(websocket);

  const persistence = createSessionPersistence(env);
  await persistence.init();

  app.get('/healthz', async () => {
    return { ok: true };
  });

  const llmGateway = createLLMGateway(env, app.log);
  const personaRag = createPersonaRag(env, app.log);
  const personaJobs = createPersonaJobManager(env, app.log);
  const transcriptStore = createTranscriptStore(env, app.log);
  const transcriptRag = createTranscriptRag(env, app.log);
  const userModelJobs = createUserModelJobManager(env, app.log);
  const gameService = createGameService(llmGateway, persistence, {
    eventLogMax: env.eventLogMax,
    historyTurns: env.llmHistoryTurns,
    logger: app.log,
    personaContextProvider: (message, options) => personaRag.getPersonaContext({ message, signal: options?.signal }),
    transcriptContextProvider: (params) => transcriptRag.getTranscriptContext(params),
    onEventPersisted: ({ sessionId, state, events, event }) => {
      void transcriptStore.appendInteraction(sessionId, event);

      if (env.userModelEnabled && state.userId && (env.userModelUpdateOnInteraction || state.outcome.ended)) {
        userModelJobs.queueUpdateFromSession({
          userId: state.userId,
          sessionId,
          state,
          events,
          lastEvent: event
        });
      }
    }
  });

  const voiceService = createVoiceService(env, app.log);

  registerHealthRoutes(app);
  registerSessionRoutes(app, gameService, { personaJobs });
  registerInteractionRoutes(app, gameService);
  registerProgressionRoutes(app, gameService);
  registerAppearanceRoutes(app, gameService);
  registerOutreachRoutes(app, gameService);
  registerStreamRoutes(app, gameService);
  registerPersonaRoutes(app, { personaJobs, personaRag });
  registerTranscriptRoutes(app, env);
  registerUserRoutes(app, { env, gameService, userModelJobs });
  registerVoiceRoutes(app, voiceService);
  registerConvaiRoutes(app, gameService, env);

  await app.listen({ port: env.port, host: '0.0.0.0' });
  app.log.info(`social pet api running at http://localhost:${env.port}`);
}

void bootstrap().catch((error) => {
  console.error(error);
  process.exit(1);
});
