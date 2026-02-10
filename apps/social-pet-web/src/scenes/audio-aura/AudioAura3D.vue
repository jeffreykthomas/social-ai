<script setup lang="ts">
import * as THREE from 'three';
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

const props = defineProps<{
  level: number;
  isListening: boolean;
  isSpeaking: boolean;
  isThinking: boolean;
  isTalking: boolean;
  height?: number;
}>();

const canvasHeight = computed(() => props.height ?? 300);

const host = ref<HTMLDivElement | null>(null);

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let ringGroup: THREE.Group | null = null;
let coreMesh: THREE.Mesh | null = null;
let frameHandle = 0;
let bars: Array<{ mesh: THREE.Mesh; angle: number; seed: number; hue: number }> = [];
let elapsed = 0;
let smoothEnergy = 0;

function initScene(): void {
  if (!host.value) return;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(42, host.value.clientWidth / canvasHeight.value, 0.1, 100);
  camera.position.set(0, 0, 14);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(host.value.clientWidth, canvasHeight.value);
  host.value.appendChild(renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 0.9);
  scene.add(ambient);

  const key = new THREE.PointLight(0xffffff, 1.2, 40);
  key.position.set(4, 6, 12);
  scene.add(key);

  const fill = new THREE.PointLight(0x7ed6df, 0.7, 40);
  fill.position.set(-6, -4, 8);
  scene.add(fill);

  ringGroup = new THREE.Group();
  scene.add(ringGroup);

  const barCount = 180;
  const radius = 3.6;
  const geometry = new THREE.BoxGeometry(0.07, 1, 0.24);

  for (let i = 0; i < barCount; i += 1) {
    const angle = (i / barCount) * Math.PI * 2;
    const hue = 170 + ((i / barCount) * 110 - 55);
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(`hsl(${hue}, 65%, 60%)`),
      metalness: 0.45,
      roughness: 0.28,
      transparent: true,
      opacity: 0.95
    });

    const mesh = new THREE.Mesh(geometry, material);
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;

    mesh.position.set(x, y, 0);
    mesh.rotation.z = angle - Math.PI / 2;

    ringGroup.add(mesh);
    bars.push({ mesh, angle, seed: Math.random() * Math.PI * 2, hue });
  }

  const coreGeometry = new THREE.SphereGeometry(2.42, 64, 64);
  const coreMaterial = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#d9f8ea'),
    metalness: 0.03,
    roughness: 0.18,
    transmission: 0.42,
    transparent: true,
    opacity: 0.95,
    thickness: 0.9
  });

  coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
  scene.add(coreMesh);

  const resizeObserver = new ResizeObserver(() => {
    if (!host.value || !renderer || !camera) return;
    const width = host.value.clientWidth;
    const height = host.value.clientHeight || canvasHeight.value;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });

  resizeObserver.observe(host.value);

  onUnmounted(() => {
    resizeObserver.disconnect();
  });
}

function animate(): void {
  if (!renderer || !scene || !camera || !ringGroup || !coreMesh) return;

  elapsed += 0.016;

  // Target energy: distinct levels for each creature state
  let targetEnergy = 0;
  if (props.isTalking || props.isSpeaking) {
    targetEnergy = 1.0;
  } else if (props.isThinking) {
    targetEnergy = 0.35;
  } else if (props.isListening) {
    targetEnergy = 0.25;
  }
  targetEnergy = Math.min(1, targetEnergy + Math.min(1, props.level) * 0.2);

  // Smooth transition — ramp up faster than decay for snappy response feel
  const lerpRate = targetEnergy > smoothEnergy ? 0.05 : 0.028;
  smoothEnergy += (targetEnergy - smoothEnergy) * lerpRate;

  const e = Math.max(0, smoothEnergy);

  for (let i = 0; i < bars.length; i += 1) {
    const entry = bars[i];

    // Idle: short bars with slow, subtle breathing
    const breath = 0.06 + Math.sin(elapsed * 0.5 + entry.seed) * 0.015;

    // Active: dramatic speech-like undulation with layered waves
    const w1 = Math.abs(Math.sin(elapsed * 8.5 + entry.seed * 2.3)) * 0.38;
    const w2 = Math.abs(Math.sin(elapsed * 14.2 + entry.angle * 3.5 + entry.seed * 0.8)) * 0.28;
    const w3 = Math.sin(elapsed * 3.5 + i * 0.1) * 0.1;
    const speech = 0.34 + w1 + w2 + w3;

    // Blend idle → active based on smooth energy
    const height = breath + e * (speech - breath);

    const radialX = Math.cos(entry.angle);
    const radialY = Math.sin(entry.angle);
    const ringScale = 0.38 + e * 0.62;
    const radialOffset = 3.6 * ringScale + height * 0.5;

    entry.mesh.scale.y = height;
    entry.mesh.position.x = radialX * radialOffset;
    entry.mesh.position.y = radialY * radialOffset;

    // Colors: muted when idle, vivid when active
    const mat = entry.mesh.material as THREE.MeshStandardMaterial;
    const sat = 42 + e * 32;
    const light = 46 + e * 18;
    mat.color.set(`hsl(${entry.hue}, ${sat}%, ${light}%)`);
    mat.emissive.set(`hsl(${entry.hue}, 38%, ${Math.min(42, 2 + e * 38)}%)`);
    mat.emissiveIntensity = 0.04 + e * 1.15;
    mat.opacity = 0.5 + e * 0.45;
  }

  // Ring rotation: very slow when idle, energetic when active
  ringGroup.rotation.z += 0.0006 + e * 0.005;
  ringGroup.rotation.x = Math.sin(elapsed * 0.22) * (0.015 + e * 0.09);

  // Core sphere: small when idle, expands to full size when speaking
  const coreIdleScale = 0.38 + Math.sin(elapsed * 0.8) * 0.008;
  const coreActiveScale = 1.0 + Math.sin(elapsed * 2.8) * 0.03;
  const coreScale = coreIdleScale + e * (coreActiveScale - coreIdleScale);
  coreMesh.scale.set(coreScale, coreScale, coreScale);
  coreMesh.rotation.y += 0.0008 + e * 0.0025;

  renderer.render(scene, camera);
  frameHandle = requestAnimationFrame(animate);
}

onMounted(() => {
  initScene();
  animate();
});

onUnmounted(() => {
  cancelAnimationFrame(frameHandle);

  if (ringGroup) {
    ringGroup.traverse((obj: THREE.Object3D) => {
      if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry.dispose();
      const material = (obj as THREE.Mesh).material;
      if (Array.isArray(material)) {
        material.forEach((m) => m.dispose());
      } else if (material) {
        material.dispose();
      }
    });
  }

  if (coreMesh) {
    coreMesh.geometry.dispose();
    const mat = coreMesh.material;
    if (Array.isArray(mat)) {
      mat.forEach((m) => m.dispose());
    } else {
      mat.dispose();
    }
  }

  renderer?.dispose();

  renderer = null;
  scene = null;
  camera = null;
  ringGroup = null;
  coreMesh = null;
  bars = [];
  smoothEnergy = 0;
});

watch(canvasHeight, () => {
  if (!host.value || !renderer || !camera) return;
  const width = host.value.clientWidth;
  renderer.setSize(width, canvasHeight.value);
  camera.aspect = width / canvasHeight.value;
  camera.updateProjectionMatrix();
});

watch(
  () => [props.level, props.isListening, props.isSpeaking, props.isThinking, props.isTalking],
  () => {
    // animation loop consumes latest props directly
  }
);
</script>

<template>
  <div class="audio-aura-shell">
    <div ref="host" class="audio-aura-canvas" :style="{ height: canvasHeight + 'px' }" />
  </div>
</template>

<style scoped>
.audio-aura-shell {
  margin: 14px 16px 0;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid #d4e4ef;
  background: radial-gradient(circle at 30% 15%, #b8f5e2 0%, #8ed6e8 28%, #f4daac 58%, #ffd3dc 100%);
}

.audio-aura-canvas {
  width: 100%;
  /* height driven dynamically via :style binding from height prop */
}
</style>
