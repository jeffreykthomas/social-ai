# Vue + Node (TypeScript) Social-Tamagotchi: Architecture Plan (v0)

## Amendment Notice

This base plan is amended by `/Users/jeffreythomas/Documents/social-ai/docs/social-creature-design-plan-amendment-v2.md`.
For gameplay scope, pacing, assessment, failure state, and monetization, treat the amendment as the source of truth.

## 1) Product Intent

Build a web app game where the user raises an AI creature through social interaction (not physical caretaking).  
The companion is an adult the whole way through, progressing through life phases (young adult -> middle aged -> wise -> old).

Core goals:
- Teach human-relevant social skills through interaction.
- Preserve deep character variance across runs.
- Support model backend switching between OpenAI and Anthropic.
- Stay compatible with this repo's social-agent direction and `model/character.json`.

## 2) Experience Pillars

- Social needs loop: Player actions change the creature's social state.
- Development loop: Repeated interaction shapes long-term traits and appearance.
- Narrative loop: Journey follows a hero's-journey-style arc.
- Reflective learning: The game explains what interaction patterns helped or hurt.

## 3) Game Arc (Hero's Journey Mapping)

Use a three-act state machine:

- Act I: Safe Bonding (early game)
  - Creature is trusting, expressive, and mostly easy to satisfy.
  - Goal: establish attachment and communication rhythm.
- Act II: Trials and Friction (mid game)
  - Stressors emerge: misunderstandings, boundary tests, conflicting needs.
  - Goal: teach repair, empathy, boundaries, and conflict skills.
- Act III: Integration (end game)
  - Creature stabilizes into adult identity and social style.
  - Ending summarizes who it became and why.

Progression triggers:
- Maturity score thresholds
- Relationship stability history
- Conflict-repair performance over time

## 4) Domain Model

### 4.1 Runtime State (fast-changing)
- `SocialNeedState`
  - `connection`, `safety`, `approval`, `empathy`, `autonomy`, `fun`, etc.
  - values in `[0,1]` with decay and event-based deltas
- `MoodState`
  - short-lived affect and arousal
- `BondState`
  - trust, attachment security, rupture-repair balance
- `StageState`
  - young adult / middle aged / wise / old life-phase mode
- `NarrativeState`
  - current act, trial counters, active dilemmas

### 4.2 Identity State (slow-changing)
- `CharacterGenome` (seed + latent factors)
  - maps to stable trait vectors
- `CharacterProfile`
  - selected fields from `model/character.json`
- `GrowthTrajectory`
  - record of trait drifts caused by repeated interactions

### 4.3 Event State
- `InteractionEvent`
  - user input, inferred intent, model response, need deltas
- `MilestoneEvent`
  - stage shifts, conflicts, resolved arcs
- `AppearanceEvent`
  - evolution choice / cosmetic unlock based on traits

## 5) Mapping `model/character.json` to Gameplay

`model/character.json` is broad; for v0, use a curated subset:

- Must-use categories:
  - `motivations_needs_drives`
  - `values_beliefs_ethics`
  - `social_style_relationship_patterns`
  - `emotion_affect_regulation`
  - `communication_voice_expression`
  - `attachment_intimacy_boundaries`
  - `conflict_power_strategy`
  - `growth_arc_change_dynamics`
  - `agent_operational_controls`
- Optional later:
  - demographics/culture/resources/health/aesthetic layers

Implementation approach:
- Keep a `character_core.json` per creature (derived from full schema).
- Keep a `character_live_state.json` for runtime deltas.
- Never rewrite immutable identity anchors; only adjust growth-capable fields.

## 6) Frontend Architecture (Vue + TypeScript)

Use Vue 3 + Vite + TypeScript + Pinia.

For the character scene:
- Start with 2D canvas via PixiJS (or Phaser if you prefer game-first ergonomics).
- Keep game logic in domain services, not in scene components.

Recommended project structure:

```text
apps/social-pet-web/
  src/
    app/
      main.ts
      router.ts
      providers.ts
    core/
      config/
      errors/
      http/
      logging/
      persistence/
    features/
      onboarding/
      pet-core/
      interaction/
      progression/
      narrative/
      appearance/
      settings/
    shared/
      components/
      models/
      utils/
    scenes/
      pet-room/
```

Layering per feature:
- Presentation: views/components/composables
- Application: use-cases (`ProcessInteraction`, `AdvanceNarrativeAct`, etc.)
- Domain: entities/value objects/rules
- Data: repositories/datasources/mappers

State management:
- Pinia stores + typed composables for feature states
- Event queue for interaction ticks to keep deterministic progression

## 7) Backend Architecture (Node + TypeScript)

Use a Node service as authoritative backend:
- Fastify + TypeScript (lean) or NestJS (more structure).
- PostgreSQL for persistent state.
- Redis for short-lived interaction sessions / queues.

Suggested backend package layout:

```text
services/social-pet-api/
  src/
    app.ts
    modules/
      session/
      interaction/
      progression/
      narrative/
      appearance/
      analytics/
    domain/
      entities/
      rules/
      services/
    infra/
      db/
      cache/
      providers/
```

Hard rule:
- Backend owns all canonical progression state and provider keys.
- Frontend is a client and local cache only.

## 8) LLM Provider Switching (OpenAI/Anthropic)

Use a provider-agnostic gateway interface:

```text
LLMGateway
  -> OpenAIAdapter
  -> AnthropicAdapter
```

Routing rules:
- Global default provider in remote config
- Per-feature override (e.g., narrative vs dialogue)
- Fallback chain on failure/timeout

Important security boundary:
- Browser app should not hold provider API keys directly.
- Use a backend "Model Gateway API" service for signed requests, quotas, telemetry, and policy checks.

## 9) API Boundary (Thin Service First)

Create a thin JSON API first:

- `POST /session/start`
- `POST /interaction/respond`
- `POST /progression/tick`
- `POST /appearance/render_prompt`
- `GET /session/{id}/state`

Responsibilities:
- Manage provider adapters (OpenAI/Anthropic)
- Maintain authoritative game state and snapshots
- Apply deterministic rule engine before/after model calls
- Store replay logs for debugging and balancing

## 10) Core Game Loop

Per interaction tick:
1. Read current runtime state.
2. Parse user input into social action candidates.
3. Compute deterministic need impact priors.
4. Ask LLM for response + inferred emotional interpretation.
5. Reconcile model output with safety/rule constraints.
6. Apply state transitions (needs, trust, stage, narrative).
7. Emit UI update + save event log.

Deterministic-first principle:
- Models generate language and soft interpretation.
- Rules decide hard transitions and progression thresholds.

## 11) Visual Evolution Strategy

Use trait-conditioned generation prompts rather than full random regeneration:

- Base creature seed locked at creation.
- Evolution checkpoints update:
  - expression set
  - posture
  - accessories/markings
  - aura/color accents
- Keep identity continuity between stages.

v0 suggestion:
- Start with handcrafted layers (eyes/body/accessory/background variants).
- Add AI image generation pipeline later to avoid early art instability.

## 12) Safety and Learning Design

- Disallow manipulative reward loops (no "perform for approval only" reinforcement).
- Reward prosocial repair behaviors:
  - active listening
  - boundary respect
  - accountability
  - conflict repair
- Add reflection cards after key interactions:
  - "What worked"
  - "What hurt"
  - "Try next time"

## 13) Telemetry and Evaluation

Track per session:
- Retention curve by progression stage
- Conflict-resolution completion rate
- Need volatility (too flat = boring, too spiky = frustrating)
- Trait divergence index across users (uniqueness KPI)
- Reflection acceptance and follow-through

## 14) Phased Build Plan

### Phase 0: Foundations (1-2 weeks)
- Initialize monorepo structure with:
  - `apps/social-pet-web` (Vue)
  - `services/social-pet-api` (Node TS)
- Add shared domain package for types and events:
  - `packages/social-pet-domain`
- Implement one-screen interaction loop with text + mood UI

### Phase 1: Core Gameplay (2-4 weeks)
- Add need engine and trust/attachment model
- Add Act I to Act II transitions
- Add provider switch in Node gateway (OpenAI/Anthropic)

### Phase 2: Narrative and Growth (3-5 weeks)
- Add hero-journey act system and trial events
- Add stage evolution and adult ending summary
- Add replay + analytics events

### Phase 3: Visual and Personalization (ongoing)
- Add richer animation and conditional appearance updates
- Add asset pipeline for AI-generated variants with moderation

## 15) Immediate Decisions Needed

- v0 interaction modality:
  - text only, or text + voice
- v0 visuals:
  - handcrafted assets first (recommended), or AI-generated from day one
- deployment mode:
  - local dev only, or cloud dev/staging from the start
- first launch target:
  - desktop web first, or responsive web for desktop + mobile
