# Social-AI Agent Instructions (All AI Agents)

This file applies to all AI agents working in this repository (Codex, Claude, Cursor, ChatGPT, etc.).

## Hard Requirements

- Be non-interactive by default: make reasonable assumptions and proceed.
- Prefer minimal diffs: change only what is required for the task.
- Use Yarn for JS/TS workspaces (`yarn`, not `npm`).
- Do not create PRs directly; leave commit/push/PR orchestration to the host workflow.
- Keep production safety in mind; avoid bypassing architecture with temporary hacks.

## Repository Scope

This repository currently contains two major tracks:

1. Social Pet app (active product surface)
- `apps/social-pet-web` (Vue 3 + Vite + TypeScript + Pinia)
- `services/social-pet-api` (Fastify + TypeScript)
- `packages/social-pet-domain` (shared domain types)

2. Legacy predictive-agent simulation stack
- `reverie/`
- `environment/`
- `chat-backbone/`

Default to the Social Pet stack unless the task explicitly targets legacy paths.

## JS/TS Workspace Conventions

- Root scripts:
  - `yarn dev:web`
  - `yarn dev:api`
  - `yarn typecheck`
  - `yarn build`
- Prefer store/service-centered logic over ad hoc UI mutation.
- Keep domain types centralized in `packages/social-pet-domain`.
- For streaming interactions, preserve low-latency behavior and cancellation semantics.

## Continuity Ledger Program (Required)

All agents must maintain a continuity ledger for the active task.

### Naming Convention

Use:
- `CONTINUITY-{agent}-{task-slug}.md`

Examples:
- `CONTINUITY-codex-streaming-interruptions.md`
- `CONTINUITY-claude-assessment-engine-v0.md`

### Setup

- If no task ledger exists, create one from `CONTINUITY.template.md`.
- Keep the active ledger at repo root for visibility.
- At the start of each turn, read the active ledger and refresh it before substantive work.

### When To Update

Update the ledger when any of these change:
- Goal/success criteria
- Constraints or assumptions
- Key decisions
- Progress state (`Done`, `Now`, `Next`)
- Important tool outcomes

### Quality Rules

- Keep entries concise and factual.
- Mark uncertainty as `UNCONFIRMED`.
- Prefer bullets and stable summaries over transcripts.
- If continuity is missing after compaction, rebuild from visible context and ask up to 1-3 targeted questions only if blocked.

### Plan Tool vs Ledger

- Use `update_plan` for short-term execution scaffolding.
- Use continuity ledger for long-running context continuity across turns/sessions/compaction.
- Keep both consistent at intent/progress level.

## Testing And Validation Expectations

For Social Pet changes:
- Always run `yarn typecheck`.
- Run `yarn build` when behavior or interfaces changed.
- If a task cannot run validation, explicitly note why.

For legacy Python/reverie changes:
- Run only targeted checks/scripts relevant to touched code.
- Do not run heavyweight long simulations unless requested.

## Documentation Hygiene

- If architecture or workflow changes materially, update docs in `docs/`.
- If agent-facing process changes, update this `AGENTS.md` and `CONTINUITY.template.md`.
