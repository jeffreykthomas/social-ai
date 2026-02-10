import assert from 'node:assert/strict';
import test from 'node:test';

import type { OutreachConsent, SessionState } from '@social-pet/domain';
import Fastify from 'fastify';

import {
  createInitialAssessmentState,
  createInitialHealthState,
  createInitialKnowledgeState,
  createInitialTimelineState
} from '../../domain/phase1Engines';
import { registerOutreachRoutes } from './routes';

function makeState(): SessionState {
  return {
    sessionId: 'sp_test',
    needs: {
      connection: 0.5,
      safety: 0.5,
      approval: 0.5,
      empathy: 0.5,
      autonomy: 0.5,
      fun: 0.5
    },
    stage: { mode: 'young_adult' },
    knowledge: createInitialKnowledgeState(),
    bond: { trust: 0.5, ruptureRepairBalance: 0.5 },
    progress: { maturity: 0.1, interactions: 1 },
    narrative: { act: 'safe_bonding', activeTrial: null },
    timeline: createInitialTimelineState('2026-02-08T00:00:00.000Z'),
    assessment: createInitialAssessmentState(),
    health: createInitialHealthState(),
    outcome: { ended: false },
    outreach: { consent: 'unknown', nudgeCount: 0 }
  };
}

test('outreach routes return nudge and update outreach state', async () => {
  const baseState = makeState();
  const app = Fastify();

  registerOutreachRoutes(
    app,
    {
      getOutreachNudge: async () => ({
        shouldNotify: true,
        severity: 'urgent',
        inactivityHours: 26,
        askForContact: true,
        message: 'Please check in.'
      }),
      updateOutreachPreference: async (
        _sessionId: string,
        preference: { consent: OutreachConsent; contactHint?: string }
      ) => ({
        ...baseState,
        outreach: {
          ...baseState.outreach,
          consent: preference.consent,
          contactHint: preference.contactHint
        }
      }),
      markOutreachSent: async () => ({
        ...baseState,
        outreach: {
          ...baseState.outreach,
          nudgeCount: 1,
          lastNudgeAt: '2026-02-08T12:00:00.000Z'
        }
      })
    } as any
  );

  await app.ready();

  const getResponse = await app.inject({
    method: 'GET',
    url: '/session/sp_test/outreach'
  });
  assert.equal(getResponse.statusCode, 200);
  assert.equal(getResponse.json().severity, 'urgent');

  const preferenceResponse = await app.inject({
    method: 'POST',
    url: '/session/sp_test/outreach/preferences',
    payload: { consent: 'granted', contactHint: 'push-ok' }
  });
  assert.equal(preferenceResponse.statusCode, 200);
  assert.equal(preferenceResponse.json().state.outreach.consent, 'granted');

  const markSentResponse = await app.inject({
    method: 'POST',
    url: '/session/sp_test/outreach/mark-sent',
    payload: {}
  });
  assert.equal(markSentResponse.statusCode, 200);
  assert.equal(markSentResponse.json().state.outreach.nudgeCount, 1);

  await app.close();
});

test('outreach routes return 404 when session is missing', async () => {
  const app = Fastify();

  registerOutreachRoutes(
    app,
    {
      getOutreachNudge: async () => null,
      updateOutreachPreference: async () => null,
      markOutreachSent: async () => null
    } as any
  );

  await app.ready();

  const getResponse = await app.inject({
    method: 'GET',
    url: '/session/unknown/outreach'
  });
  assert.equal(getResponse.statusCode, 404);

  const preferencesResponse = await app.inject({
    method: 'POST',
    url: '/session/unknown/outreach/preferences',
    payload: { consent: 'declined' }
  });
  assert.equal(preferencesResponse.statusCode, 404);

  const markResponse = await app.inject({
    method: 'POST',
    url: '/session/unknown/outreach/mark-sent',
    payload: {}
  });
  assert.equal(markResponse.statusCode, 404);

  await app.close();
});
