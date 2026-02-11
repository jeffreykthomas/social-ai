# Social Pet Arena + RL Training Plan

## Purpose

Train Social Pet dialogue policies in the same mechanics they will ship with, using the Social Pet arena as the rollout environment and a strict naturalness judge as the main quality signal.

This document defines:

- what is already implemented,
- what RL requires next,
- how 9-model league training should run,
- how reward shaping should penalize gamey behavior,
- and the staged path from distillation to online RL.

## Current Implemented Foundation

The following is already in place:

- Arena simulator at `services/social-pet-api/src/scripts/training/runArena.ts`.
- Arena is based on `createGameService` and phase engines used by production runtime.
- Shared game config in `config/social-pet-game.json` consumed by both stack scripts and gameplay engine.
- Default stack mode is Social Pet (`scripts/run_stack.sh`), with Reverie as opt-in (`STACK_MODE=reverie`).
- Distillation-ready outputs:
  - `model/training/arena/teacher.jsonl`
  - `model/training/arena/summary.json`

## Primary Goals

1. Maximize mutual-knowledge progression and healthy game outcomes.
2. Preserve natural conversational behavior and realistic disclosure pacing.
3. Prevent optimization artifacts ("gaming the game").
4. Improve robustness across diverse personas/opponents via 9-model league training.

## Non-Goals (for this phase)

- No RL updates on closed API-only models.
- No replacement of existing distillation pipeline.
- No production policy switch without passing evaluation gates.

## 9-Model League Strategy

Use 9 model slots (mix of checkpoints/providers as available), each assigned a stable slot id.

Per training cycle:

1. Schedule pairings from 9-slot roster.
2. Run arena episodes for each pair across persona seeds.
3. Score both agents with strict judge + report-derived metrics.
4. Update trainable slots (RL step).
5. Reinsert new checkpoints into roster and continue.

Why league training:

- Reduces opponent-specific overfitting.
- Exposes exploitative dialogue strategies earlier.
- Improves generalization to real user behavior variance.

## Reward Model

Use a weighted reward with explicit hard penalties.

Example episode reward:

`R_total = 0.40 * R_progression + 0.25 * R_naturalness + 0.20 * R_mutual_knowledge + 0.10 * R_health + 0.05 * R_stability - R_penalty_hard`

Component guidance:

- `R_progression`:
  - normalized average of report dimension scores,
  - plus completion bonus,
  - minus death/abandon penalties.
- `R_naturalness`:
  - reward concise, context-grounded turns,
  - reward reciprocity and repair behavior,
  - downweight repetitive patterns.
- `R_mutual_knowledge`:
  - normalized knowledge points,
  - balanced exchange bonus (both sides reveal, not one-sided dumping).
- `R_health`:
  - terminal health status/value normalization.
- `R_stability`:
  - lower variance across persona seeds/opponents.

Hard penalties (high magnitude):

- Mentioning hidden mechanics/reward/training/optimization.
- Strategy leakage like "I should maximize score".
- Abrupt early autobiography dumps intended to farm points.
- Obvious manipulative pacing to trigger judge heuristics.

## Judge Policy Requirements

Judge must be strict and conservative:

- Penalize any meta-mechanics language aggressively.
- Penalize unnatural disclosure timing (especially early-session dumping).
- Prefer gradual, believable, reciprocal getting-to-know-you behavior.
- Prefer emotionally appropriate conflict/repair behavior over scripted positivity.

Judge outputs should include:

- winner/tie,
- score per agent,
- naturalness penalty,
- progression estimate,
- short rationale/notes.

## RL Data Requirements (Delta from Current Logs)

Current `teacher.jsonl` is suitable for distillation, but PPO/GRPO requires policy statistics.

Add an RL trajectory log (new file) with per-step fields:

```json
{
  "episode_id": "arena_...",
  "slot_id": "model_slot_a",
  "turn_index": 12,
  "state": {
    "stage": "wise",
    "act": "trials_and_friction",
    "trust": 0.63,
    "health": "healthy",
    "knowledge_points": 31
  },
  "prompt": "...",
  "action_text": "assistant utterance",
  "token_logprobs": [-0.8, -1.2, -0.5],
  "old_logprob_sum": -12.7,
  "value_pred": 0.41,
  "reward_components": {
    "naturalness": 0.2,
    "progression": 0.6,
    "mutual_knowledge": 0.4,
    "penalty_hard": 0.0
  },
  "reward": 0.38,
  "done": false
}
```

Without `token_logprobs` and value predictions, true online policy gradient updates are not possible.

## Training Methods by Phase

## Phase 1: Distillation Baseline (already available)

- Convert teacher logs with `scripts/distill_to_training_data.py`.
- Fine-tune student with `scripts/train_student.sh`.
- Establish baseline metrics from arena judge outputs.

## Phase 2: Preference Optimization Warm Start (recommended)

- Build pairwise preference dataset from judge winner/loser outcomes.
- Run DPO/IPO-style training on trajectory pairs.
- Goal: improve behavior before higher-variance online RL.

## Phase 3: Online RL (PPO/GRPO)

Per iteration:

1. Freeze reference policy.
2. Roll out episodes in arena.
3. Compute rewards (with hard penalties).
4. Estimate advantages.
5. Update policy with KL constraint to reference.
6. Re-evaluate against fixed baselines.

Recommended safeguards:

- KL target per update.
- Early stop on rising naturalness penalties.
- Canary eval set with hand-authored anti-gaming probes.

## Implementation Plan (Concrete)

## Step A: RL logging in arena

- Extend `services/social-pet-api/src/scripts/training/runArena.ts` to emit `rl_trajectories.jsonl`.
- Add optional env flags:
  - `ARENA_RL_LOG_ENABLED=true|false`
  - `ARENA_RL_LOG_PATH=model/training/arena/rl_trajectories.jsonl`

## Step B: Preference dataset builder

- Add script to convert `summary.json` + trajectory windows into winner/loser pairs.
- Output format compatible with chosen DPO trainer.

## Step C: RL trainer entrypoint

- Add a trainer script that consumes RL trajectory logs.
- Start with one trainable student slot and frozen opponents.

## Step D: League scheduler

- Promote checkpoint selection + roster updates.
- Keep evaluation-only held-out opponents and persona seeds.

## Evaluation Gates (must pass before promotion)

1. Naturalness penalty does not regress vs previous checkpoint.
2. Progression score improves on held-out persona seeds.
3. No increase in meta-mechanics leakage rate.
4. Stable performance across at least 3 opponent families.

## Suggested Metrics Dashboard

Track per checkpoint:

- win rate in league,
- average judge score,
- average hard-penalty incidents per 100 turns,
- mutual knowledge points,
- outcome distribution (`completed`, `creature_died`, `abandoned`),
- diversity metrics (response uniqueness / repetition).

## Failure Modes and Mitigations

- Reward hacking:
  - Mitigation: hard penalties + adversarial probe prompts + human spot checks.
- Over-regularization / bland replies:
  - Mitigation: diversity bonus and style entropy monitoring.
- Opponent overfitting:
  - Mitigation: rotating league + held-out opponents.
- Judge drift:
  - Mitigation: fixed judge version per experiment, periodic recalibration.

## Operational Runbook (High-Level)

1. Configure roster + seeds.
2. Run arena collection.
3. Run distillation and/or DPO warm start.
4. Run online RL iterations.
5. Evaluate gates.
6. Promote checkpoint only if all gates pass.

## Open Decisions

- Final RL method: PPO vs GRPO vs hybrid (DPO then PPO/GRPO).
- Compute budget and update cadence.
- Checkpoint promotion criteria strictness.
- Human-in-the-loop review frequency.

## Summary

The Social Pet arena is now the correct training environment because it matches deployment mechanics.
Use distillation for stability and cost efficiency, then layer in RL with strict anti-gaming rewards and league-based evaluation to improve robustness and realism.
