import assert from 'node:assert/strict';
import test from 'node:test';

import type { SessionOutcomeState } from '@social-pet/domain';

import {
  applyHealthDayAdvance,
  createInitialHealthState,
  createInitialTimelineState,
  deriveOutcome,
  evaluateOutreachNudge,
  wholeDaysBetween
} from './phase1Engines';

test('wholeDaysBetween returns full elapsed day count', () => {
  const start = '2026-02-01T00:00:00.000Z';
  assert.equal(wholeDaysBetween(start, '2026-02-01T23:59:59.000Z'), 0);
  assert.equal(wholeDaysBetween(start, '2026-02-02T00:00:00.000Z'), 1);
  assert.equal(wholeDaysBetween(start, '2026-02-03T00:00:00.000Z'), 2);
});

test('applyHealthDayAdvance compounds neglect after first missed day grace period', () => {
  const initial = createInitialHealthState();

  const day1 = applyHealthDayAdvance(initial, false);
  assert.equal(day1.delta, 0);
  assert.equal(day1.health.consecutiveMissedDays, 1);
  assert.equal(day1.health.value, 80);

  const day2 = applyHealthDayAdvance(day1.health, false);
  assert.equal(day2.delta, -5);
  assert.equal(day2.health.consecutiveMissedDays, 2);
  assert.equal(day2.health.value, 75);

  const day3 = applyHealthDayAdvance(day2.health, false);
  assert.equal(day3.delta, -7);
  assert.equal(day3.health.consecutiveMissedDays, 3);
  assert.equal(day3.health.value, 68);
});

test('evaluateOutreachNudge asks for contact only when consent is unknown', () => {
  const now = '2026-02-08T12:00:00.000Z';
  const thirteenHoursAgo = '2026-02-07T23:00:00.000Z';
  const thirtyHoursAgo = '2026-02-07T06:00:00.000Z';

  const gentle = evaluateOutreachNudge({
    nowIso: now,
    dayStartedAt: thirteenHoursAgo,
    healthStatus: 'healthy',
    healthValue: 80,
    consent: 'unknown',
    outcomeEnded: false
  });
  assert.equal(gentle.shouldNotify, true);
  assert.equal(gentle.severity, 'gentle');
  assert.equal(gentle.askForContact, true);

  const urgent = evaluateOutreachNudge({
    nowIso: now,
    dayStartedAt: thirtyHoursAgo,
    healthStatus: 'healthy',
    healthValue: 70,
    consent: 'granted',
    outcomeEnded: false
  });
  assert.equal(urgent.shouldNotify, true);
  assert.equal(urgent.severity, 'urgent');
  assert.equal(urgent.askForContact, false);
});

test('deriveOutcome ends run when creature dies or final day criteria are met', () => {
  const now = '2026-02-08T12:00:00.000Z';
  const ongoing: SessionOutcomeState = { ended: false };

  const dead = deriveOutcome(ongoing, now, 'wise', createInitialTimelineState(now), 'dead', 20);
  assert.equal(dead.ended, true);
  assert.equal(dead.reason, 'creature_died');

  const timeline = { ...createInitialTimelineState(now), currentDay: 14, totalDays: 14 };
  const completed = deriveOutcome(ongoing, now, 'old', timeline, 'healthy', 45);
  assert.equal(completed.ended, true);
  assert.equal(completed.reason, 'completed');
});
