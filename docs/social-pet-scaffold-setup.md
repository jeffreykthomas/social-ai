# Social Pet Scaffold Setup (Yarn)

## Workspace layout

- `apps/social-pet-web` - Vue 3 + Vite + TypeScript + Pinia + Three.js 3D audio aura
- `services/social-pet-api` - Fastify + TypeScript API (OpenAI/Anthropic adapters + WS streaming)
- `packages/social-pet-domain` - shared domain types

## Install

From repo root:

```bash
yarn install
```

## Configure env

- Copy `apps/social-pet-web/.env.example` to `apps/social-pet-web/.env`.
- Copy `services/social-pet-api/.env.example` to `services/social-pet-api/.env`.

For low latency:
- Set `LLM_TIMEOUT_MS` to a strict budget (e.g. `600-1200`).
- Use `PERSISTENCE_MODE=hybrid` with Redis + Postgres for fast reads and durable writes.
- Keep `LLM_HISTORY_TURNS` modest (e.g. `6-12`) to reduce token + latency overhead.
- Set `CORS_ORIGINS` for the active web origin (for local: `http://localhost:5173`).

## Run

Start API:

```bash
yarn dev:api
```

Start web app:

```bash
yarn dev:web
```

Web app defaults to `http://localhost:5173` and API defaults to `http://localhost:3001`.

## Interaction Modes (MVP)

The web app supports:
- `Text -> Text`
- `Text -> Voice`
- `Voice -> Text`
- `Voice -> Voice`

Voice features use browser APIs:
- Speech input: Web Speech Recognition API (`SpeechRecognition` / `webkitSpeechRecognition`)
- Speech output: `window.speechSynthesis`

For best support, use Chrome-based browsers.

## API notes

- Uses OpenAI **Responses API** when `LLM_PROVIDER=openai`.
- Uses Anthropic **Messages API** when `LLM_PROVIDER=anthropic`.
- If provider call fails or exceeds timeout budget, it returns a heuristic fallback response.
- Session state writes are async so message round-trip is not blocked by Postgres writes.
- Streaming endpoint is WebSocket at `ws://localhost:3001/interaction/stream/ws`.
- Streaming supports:
  - `user_message` to start/replace active generation.
  - `stop_stream` to cancel in-flight generation explicitly from the client.
