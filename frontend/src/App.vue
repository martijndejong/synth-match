<script setup>
import KnobComponent from './components/KnobComponent.vue';
</script>

<template>
  <main>
    <h2>Hi, test this will be the awesome synth interface</h2>
    <div class="button-row">
      <div class="knob-item">
        <KnobComponent id="Cutoff-Frequency" />
        <span class="knob-label">Cutoff Frequency</span>
      </div>
      <div class="knob-item">
        <KnobComponent id="Attack" />
        <span class="knob-label">Attack</span>
      </div>
      <div class="knob-item">
        <KnobComponent id="Decay" />
        <span class="knob-label">Decay</span>
      </div>
      <div class="knob-item">
        <KnobComponent id="Sustain" />
        <span class="knob-label">Sustain</span>
      </div>
      <div class="knob-item">
        <KnobComponent id="Release" />
        <span class="knob-label">Release</span>
      </div>
    </div>
  </main>
</template>

<style scoped>
main {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.button-row {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  max-width: 100%;
  flex-wrap: wrap;
  justify-content: center;
}

.knob-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.knob-label {
  font-size: 1rem;
  margin-top: 0.5rem;
  font-weight: bold;
}
</style>
