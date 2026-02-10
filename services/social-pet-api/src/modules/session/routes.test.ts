import assert from 'node:assert/strict';
import test from 'node:test';

import type { SessionRunReport, SessionState } from '@social-pet/domain';
import Fastify from 'fastify';

import {
  createInitialAssessmentState,
  createInitialHealthState,
  createInitialKnowledgeState,
  createInitialTimelineState
} from '../../domain/phase1Engines';
import { registerSessionRoutes } from './routes';

function makeState(sessionId = 'sp_test'): SessionState {
  return {
    sessionId,
    needs: {
      connection: 0.6,
      safety: 0.7,
      approval: 0.6,
      empathy: 0.6,
      autonomy: 0.6,
      fun: 0.6
    },
    stage: { mode: 'young_adult' },
    knowledge: createInitialKnowledgeState(),
    bond: { trust: 0.6, ruptureRepairBalance: 0.55 },
    progress: { maturity: 0.12, interactions: 4 },
    narrative: { act: 'safe_bonding', activeTrial: null },
    timeline: createInitialTimelineState('2026-02-08T00:00:00.000Z'),
    assessment: createInitialAssessmentState(),
    health: createInitialHealthState(),
    outcome: { ended: false },
    outreach: { consent: 'unknown', nudgeCount: 0 }
  };
}

function makeReport(sessionId = 'sp_test'): SessionRunReport {
  return {
    sessionId,
    generatedAt: '2026-02-08T12:00:00.000Z',
    outcome: 'in_progress',
    summary: 'Run summary.',
    mutualKnowledge: {
      points: 5,
      userFactsCount: 4,
      characterFacetsRevealedCount: 1,
      topUserFactCategories: [{ category: 'preference', count: 2 }],
      characterFacetsRevealed: ['values_beliefs_ethics'],
      sampleUserFacts: []
    },
    strengths: ['warmth is high'],
    patterns: ['support exceeds conflict'],
    confidenceNotes: ['more varied examples needed'],
    dimensions: [
      { key: 'warmth', score: 7.1, confidence: 0.65, interpretation: 'strong tendency' },
      { key: 'confrontation_comfort', score: 5.4, confidence: 0.55, interpretation: 'balanced tendency' },
      { key: 'autonomy_support', score: 6.2, confidence: 0.6, interpretation: 'balanced tendency' },
      { key: 'emotional_attunement', score: 6.8, confidence: 0.62, interpretation: 'balanced tendency' }
    ]
  };
}

test('session routes expose seed interactions and run report', async () => {
  const state = makeState();
  const report = makeReport();
  const app = Fastify();

  registerSessionRoutes(
    app,
    {
      startSession: async () => ({ sessionId: state.sessionId, state, events: [] }),
      getSession: async () => ({ state, events: [] }),
      getSeedInteractions: async () => [
        {
          id: 'seed_1',
          stage: 'young_adult',
          dayRange: [1, 3],
          kind: 'probe',
          prompt: 'Can we talk?',
          measures: ['warmth'],
          tags: ['opening']
        }
      ],
      generateRunReport: async () => report
    } as any
  );

  await app.ready();

  const startResponse = await app.inject({
    method: 'POST',
    url: '/session/start',
    payload: {}
  });
  assert.equal(startResponse.statusCode, 200);
  assert.equal(startResponse.json().sessionId, state.sessionId);

  const seedResponse = await app.inject({
    method: 'GET',
    url: `/session/${state.sessionId}/seed-interactions?limit=4`
  });
  assert.equal(seedResponse.statusCode, 200);
  assert.equal(seedResponse.json().interactions.length, 1);

  const reportResponse = await app.inject({
    method: 'GET',
    url: `/session/${state.sessionId}/report`
  });
  assert.equal(reportResponse.statusCode, 200);
  assert.equal(reportResponse.json().summary, report.summary);

  await app.close();
});

test('session report and seed routes return 404 for unknown session', async () => {
  const app = Fastify();

  registerSessionRoutes(
    app,
    {
      startSession: async () => ({ sessionId: 'x', state: makeState('x'), events: [] }),
      getSession: async () => null,
      getSeedInteractions: async () => null,
      generateRunReport: async () => null
    } as any
  );

  await app.ready();

  const stateResponse = await app.inject({
    method: 'GET',
    url: '/session/missing/state'
  });
  assert.equal(stateResponse.statusCode, 404);

  const seedResponse = await app.inject({
    method: 'GET',
    url: '/session/missing/seed-interactions'
  });
  assert.equal(seedResponse.statusCode, 404);

  const reportResponse = await app.inject({
    method: 'GET',
    url: '/session/missing/report'
  });
  assert.equal(reportResponse.statusCode, 404);

  await app.close();
});
