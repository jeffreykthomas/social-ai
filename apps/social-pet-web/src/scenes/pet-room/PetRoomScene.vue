<script setup lang="ts">
import type { HeroJourneyAct, StageMode } from '@social-pet/domain';
import { Application, Container, Graphics } from 'pixi.js';
import { onMounted, onUnmounted, ref, watch } from 'vue';

const props = defineProps<{
  trust: number;
  stage: StageMode;
  act: HeroJourneyAct;
  isThinking: boolean;
  isTalking: boolean;
}>();

const host = ref<HTMLDivElement | null>(null);

let app: Application | null = null;
let rootContainer: Container | null = null;
let shadow: Graphics | null = null;
let tail: Graphics | null = null;
let body: Graphics | null = null;
let eyeLeft: Graphics | null = null;
let eyeRight: Graphics | null = null;
let mouth: Graphics | null = null;
let resizeObserver: ResizeObserver | null = null;

const motion = {
  elapsed: 0,
  x: 0,
  y: 0,
  targetX: 0,
  targetY: 0,
  lookX: 0,
  lookY: 0,
  blink: 0,
  blinkTimer: 1.8,
  talkPulse: 0,
  breathPhase: 0
};

function stageScale(stage: StageMode): number {
  if (stage === 'young_adult') return 1.0;
  if (stage === 'middle_aged') return 1.06;
  if (stage === 'wise') return 1.11;
  return 1.16;
}

function paletteByAct(act: HeroJourneyAct): { bg: number; pet: number; accent: number } {
  if (act === 'safe_bonding') return { bg: 0xf8fbff, pet: 0x96d7ff, accent: 0x4e87ff };
  if (act === 'trials_and_friction') return { bg: 0xfff8f0, pet: 0xffc183, accent: 0xf56a4d };
  return { bg: 0xf7fff8, pet: 0x9ee5b6, accent: 0x2f9c65 };
}

function chooseNewTarget(): void {
  if (!app) return;
  const w = app.screen.width;
  const h = app.screen.height;

  motion.targetX = w * (0.25 + Math.random() * 0.5);
  motion.targetY = h * (0.5 + Math.random() * 0.2);
}

function drawCreature(): void {
  if (!app || !rootContainer || !shadow || !tail || !body || !eyeLeft || !eyeRight || !mouth) return;

  const { bg, pet, accent } = paletteByAct(props.act);
  app.renderer.background.color = bg;

  const scale = stageScale(props.stage);
  const radius = 70 * scale;
  const eyeOffsetX = 22 * scale;
  const eyeOffsetY = -10 * scale;
  const eyeRadius = Math.max(4, 7 + props.trust * 4);

  const dx = motion.targetX - motion.x;
  const dy = motion.targetY - motion.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < 4) {
    chooseNewTarget();
  }

  const stageEnergy = stageScale(props.stage) - 0.75;
  const trustEnergy = 0.7 + props.trust * 0.6;
  const thinkingDrag = props.isThinking ? 0.45 : 1;
  const roamSpeed = (20 + stageEnergy * 20) * trustEnergy * thinkingDrag;

  const dt = 1 / 60;
  const maxStep = roamSpeed * dt;
  const stepRatio = dist > 0 ? Math.min(1, maxStep / dist) : 0;

  motion.x += dx * stepRatio;
  motion.y += dy * stepRatio;

  motion.lookX += ((dx / Math.max(1, dist)) * 5 - motion.lookX) * 0.06;
  motion.lookY += ((dy / Math.max(1, dist)) * 4 - motion.lookY) * 0.06;

  motion.breathPhase += 0.04 * trustEnergy * (props.isTalking ? 1.3 : 1);
  const breathing = 1 + Math.sin(motion.breathPhase) * 0.02;

  motion.blinkTimer -= dt;
  if (motion.blinkTimer <= 0) {
    motion.blink = 1;
    motion.blinkTimer = 1.6 + Math.random() * 2.1;
  }
  motion.blink = Math.max(0, motion.blink - 0.11);
  const eyelidScale = Math.max(0.12, 1 - motion.blink * 4.2);

  motion.talkPulse += props.isTalking ? 0.22 : -0.2;
  motion.talkPulse = Math.max(0, Math.min(1, motion.talkPulse));

  const baseY = motion.y + Math.sin(motion.elapsed * 0.9) * 2;
  rootContainer.position.set(motion.x, baseY);
  rootContainer.scale.set(breathing, 1 / breathing);

  shadow.clear();
  shadow.ellipse(0, radius + 16, radius * 0.68, 14).fill({ color: 0x4d6b8f, alpha: 0.2 });

  tail.clear();
  tail.setStrokeStyle({ width: 12 * scale, color: accent, cap: 'round' });
  tail.moveTo(radius * 0.52, radius * 0.05);
  tail.bezierCurveTo(radius * 0.9, radius * 0.2, radius * 0.95, -radius * 0.25, radius * 0.65, -radius * 0.28);
  tail.stroke();

  body.clear();
  body.circle(0, 0, radius).fill(pet);
  body.circle(radius * 0.45, -radius * 0.35, radius * 0.35).fill(accent);

  eyeLeft.clear();
  eyeLeft.ellipse(-eyeOffsetX + motion.lookX, eyeOffsetY + motion.lookY, eyeRadius, eyeRadius * eyelidScale).fill(0x1f2a44);

  eyeRight.clear();
  eyeRight.ellipse(eyeOffsetX + motion.lookX, eyeOffsetY + motion.lookY, eyeRadius, eyeRadius * eyelidScale).fill(0x1f2a44);

  const smile = -18 + props.trust * 40 + motion.talkPulse * 12;
  mouth.clear();
  mouth.setStrokeStyle({ width: 4, color: 0x1f2a44, cap: 'round' });
  mouth.moveTo(-20 * scale, 20 * scale);
  mouth.bezierCurveTo(
    -8 * scale,
    (20 + smile) * scale,
    8 * scale,
    (20 + smile) * scale,
    20 * scale,
    20 * scale
  );
  mouth.stroke();
}

async function setupPixi(): Promise<void> {
  if (!host.value) return;

  app = new Application();
  await app.init({
    width: Math.max(320, host.value.clientWidth),
    height: 240,
    antialias: true,
    background: 0xf8fbff
  });

  host.value.appendChild(app.canvas);

  rootContainer = new Container();
  shadow = new Graphics();
  tail = new Graphics();
  body = new Graphics();
  eyeLeft = new Graphics();
  eyeRight = new Graphics();
  mouth = new Graphics();

  rootContainer.addChild(shadow, tail, body, eyeLeft, eyeRight, mouth);
  app.stage.addChild(rootContainer);

  motion.x = app.screen.width / 2;
  motion.y = app.screen.height * 0.62;
  motion.targetX = motion.x;
  motion.targetY = motion.y;
  chooseNewTarget();

  app.ticker.add((ticker) => {
    motion.elapsed += ticker.deltaTime * 0.03;
    drawCreature();
  });

  resizeObserver = new ResizeObserver(() => {
    if (!app || !host.value) return;
    app.renderer.resize(Math.max(320, host.value.clientWidth), 240);
    chooseNewTarget();
    drawCreature();
  });

  resizeObserver.observe(host.value);
}

onMounted(async () => {
  await setupPixi();
});

onUnmounted(() => {
  resizeObserver?.disconnect();
  resizeObserver = null;
  app?.destroy(true, { children: true });
  app = null;
  rootContainer = null;
  shadow = null;
  tail = null;
  body = null;
  eyeLeft = null;
  eyeRight = null;
  mouth = null;
});

watch(
  () => [props.trust, props.stage, props.act, props.isThinking, props.isTalking],
  () => {
    drawCreature();
  }
);
</script>

<template>
  <div class="pet-scene-shell">
    <div ref="host" class="pet-scene-canvas" />
    <div class="pet-scene-meta">
      <p><strong>Act:</strong> {{ act }}</p>
      <p><strong>Stage:</strong> {{ stage }}</p>
      <p><strong>Trust:</strong> {{ trust.toFixed(2) }}</p>
      <p><strong>Mode:</strong> {{ isTalking ? 'talking' : isThinking ? 'thinking' : 'idle' }}</p>
    </div>
  </div>
</template>