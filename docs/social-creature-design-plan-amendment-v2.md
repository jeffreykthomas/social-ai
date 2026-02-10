# Social Creature - Design Plan Amendment v2

This document amends `/Users/jeffreythomas/Documents/social-ai/docs/vue-node-social-tamagotchi-architecture.md`.
If there is conflict, this amendment takes precedence for product and gameplay scope.

## Revised Constraints

| Parameter | Original Assumption | Revised |
| --- | --- | --- |
| Total play time | ~15 hours over 8-10 weeks | ~2 hours over 2 weeks |
| Session length | 2-5 min, 3x/day | ~10 min/day |
| Total interactions | ~180 | ~50-80 |
| Failure state | Creature cannot die | Creature can wither and die if mistreated |
| Multiplayer | Open question | No (strict 1-1) |
| Post-game | Open question | Previous characters remain interactive |
| Monetization | Open question | First play free, monetize after |
| Output | Developed character | Developed character + player personality report |
| Difficulty | Open question | Single adult difficulty |

## 1) Revised Timeline And Session Shape

Baseline math:
- 14 days x ~10 minutes/day = ~140 total minutes.
- ~3-5 interaction cycles/session.
- ~50-80 interaction cycles total.

Stage cadence:
- Day 1-3: Young adult (~9-15 interactions).
- Day 4-7: Middle aged (~12-20 interactions).
- Day 8-11: Wise (~12-20 interactions).
- Day 12-14: Old/Closing (~9-15 interactions).

Design implications:
- Stage transitions are primarily time-gated with minimum interaction thresholds.
- Every session must produce meaningful progression (no filler).
- Hero's journey is compressed: crisis around Day 8-9, resolution by Day 11-12.

Daily session structure (~10 minutes):
1. Opening (1-2 min): greeting, elapsed-time reflection, visible mood/state.
2. Core interaction (5-7 min): 2-3 cycles (bid -> player response -> creature reaction/state update).
3. Closing (1-2 min): wrap-up, teaser for next session, visual growth cue if applicable.

## 2) Player Personality Assessment (Core Product Layer)

The game doubles as a covert interaction-pattern assessment:
- Creature development is the engagement mechanic.
- Personality report is the post-game deliverable.

Assessment principles:
- No explicit "test" framing during gameplay.
- Behavior-derived signals, not self-report only.
- Observable in ~80 interactions.
- Report language is insightful, not clinical diagnosis.

Target dimensions:
- Interpersonal style: warmth, directness, leading/following, consistency.
- Conflict approach: confrontation comfort, firmness/flexibility, repair initiation, fairness orientation.
- Emotional engagement: attunement, emotional vocabulary, vulnerability comfort, negative-emotion tolerance.
- Nurturing style: autonomy support/control, challenge/protection, patience, teaching approach.
- Self-revelation: openness, reciprocity, boundaries, authenticity signals.

Mechanics:
- Every interaction maps to measurable dimensions.
- Choice responses can use deterministic pre-scored mappings.
- Free-text responses use evaluator model calls that output JSON scores.
- Confidence starts low and tightens through repeated evidence.

Confidence progression target:
- Day 1-3: low confidence baseline.
- Day 4-7: medium confidence patterns.
- Day 8-11: high confidence via conflict-heavy evidence.
- Day 12-14: targeted probes + final integration.

Late-game probe planning:
- Identify dimensions below confidence threshold.
- Prioritize probe interactions for uncertain dimensions when sessions remaining are low.

## 3) Personality Report Output

Report should be generated on adulthood completion or death outcome.

Proposed sections:
1. Interaction portrait (dimension summary visual + narrative).
2. Strengths (specific interaction citations).
3. Patterns (recurring response motifs).
4. Growth edges (non-judgmental alternative approaches).
5. Creature story (how player patterns shaped this creature's trajectory).
6. Optional aggregate comparison (opt-in only).

Depth scaling by engagement:
- Minimal (<30 interactions): summary with fewer dimensions.
- Moderate (30-60): full coverage with concrete examples.
- High (60+): nuanced patterns, contradictions, growth trajectory.

Mutual knowledge as progression:
- Character -> player knowledge: trait reveals unlock as trust grows.
- Player -> character knowledge: assessment confidence improves over time.
- Stage transition input includes combined mutual-knowledge score.

## 4) Failure State: Withering And Death

Health model:
- Range 0-100, starts near 80.
- Healthy: 60-100.
- Wilting: 40-59.
- Withering: 20-39.
- Dying: 1-19.
- Dead: 0.

Health signals:
- Positive interactions/repair/consistency raise health.
- Harmful patterns/neglect/cruelty reduce health.
- Decline is visible and gradual; recovery remains possible above 0.
- Recovery difficulty increases as health state worsens.

Death handling:
- Permanent for that creature.
- Preserve history + report.
- Tone of report remains compassionate, specific, and consequence-aware.

## 5) Monetization Amendment

Base:
- First full playthrough free.
- Includes core game + basic report + saved creature.

Post-first-play monetization:
- Additional creatures (first few free, then subscription or per-creature).
- Enhanced reports.
- Creature archive and revisit depth.
- Longitudinal personality trend tracking.
- Cosmetics.

Value framing:
- Not "play more game loops" only.
- "Learn more about your own relationship patterns across creatures and contexts."

## 6) Post-Game Creature Persistence

Graduated (adult) creatures:
- Remain available for lightweight catch-up chat.
- Personality snapshot fixed at graduation.
- Shared history references remain active.
- No further progression mechanics.

Dead creatures:
- Memorialized state.
- History/report remain accessible.
- No further interaction.

## 7) Interaction Design Rules For Assessment

Rules:
- Diagnostic moments must feel natural, not quiz-like.
- Repeated measurement across changing contexts is required for confidence.
- Use buttons for clean measurement and free text for richer signal.
- Creature-initiated reciprocal questions are a major probe channel.

Pipeline:
1. Player response.
2. Structured scoring (deterministic if choice-based).
3. AI evaluation (if free-text).
4. Profile update (score + confidence + pattern extraction).
5. Probe planning for uncertain dimensions.

## 8) Cost Model (Planning Baseline)

Per game target order-of-magnitude (midpoint ~60 interactions):
- Character responses + evaluator calls + audits + transitions + images + final report.
- Combined planning baseline around low single-digit USD per full game.

Operational note:
- Revisit interactions should be low-cost because they skip progression mechanics.

## 9) Architecture Implications

Add a dedicated Assessment Engine with:
- Dimension Tracker (scores, confidence, evidence points).
- Pattern Recognizer (cross-session motifs, consistency, anomaly checks).
- Probe Planner (target uncertain dimensions with remaining-session awareness).
- Report Generator (narrative synthesis + citations + optional comparisons).

Add/extend data models:
- User assessment profile with per-dimension score/confidence/evidence.
- Creature lifecycle registry (active, graduated, dead).
- Health history + cause-coded deltas.
- Report history and longitudinal trend snapshots.

## 10) Revised Delivery Phases

Phase 1 (Core loop + basic assessment):
- Compressed 4-stage flow.
- Pre-scored choice diagnostics.
- Basic health/wilting.
- Text-only creature.

Phase 2 (Free text + full assessment):
- Free-text evaluator scoring.
- Full dimension set + probe planner.
- Basic report generation.
- Death mechanic enabled.

Phase 3 (Narrative + visuals):
- Compressed hero's journey crisis/resolution.
- Visual evolution moments and withering/death visuals.
- Improved report quality and citations.

Phase 4 (Persistence + monetization):
- Graduation/archive/revisit flows.
- Multi-creature support.
- Payment/subscription surfaces.
- Longitudinal insights.

Phase 5 (Polish + launch):
- Onboarding and tutorial tuning.
- Report UX refinement.
- Balance/playtest iteration.
- Analytics hardening.

## 11) Key Risks And Mitigations

Risk: gameplay feels like a test.
- Mitigation: keep assessment invisible, avoid score UI, natural scenario design.

Risk: 2 hours may underpower signal quality.
- Mitigation: fewer dimensions with stronger confidence, aggressive probe planning, confidence transparency.

Risk: death feels punitive.
- Mitigation: gradual withering, clear warnings, recoverability before zero, compassionate report framing.

Risk: players optimize for "right answers."
- Mitigation: multi-valid outcomes, free-text emphasis, cross-session consistency checks.

Risk: weak validity claims.
- Mitigation: ground dimensions in established frameworks, position as reflective insight not diagnosis.
