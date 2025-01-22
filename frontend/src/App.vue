<script setup>
import { ref, onMounted } from 'vue'
import KnobComponent from './components/KnobComponent.vue';
import AudioPlayer from './components/AudioPlayer.vue'; // Import the AudioPlayer component
import ADSRChart from './components/ADSRChart.vue'; // Import the new ADSRChart
import LoggingComponent from './components/LoggingComponent.vue' // Import LoggingComponent
import MatchKnobs from './components/MatchKnobs.vue'; // Import KnobControls
import { loadKnobsFromYaml } from './utils/loadKnobs';   // Import utility function


// Reactive array to store knob values
const knobValues = ref({
  'Cutoff-Frequency': 0,
  'Attack': 0,
  'Decay': 0,
  'Sustain': 0,
  'Release': 0,
});

const knobs = [
        {
          id: 'filter_cutoff_frequency',
          display_name: 'Cutoff Frequency',
          min: 20,
          max: 20000,
          scaling_factor: 1,
          default_value: 1000,
          step: 1,
          scale_type: 'logarithmic',
        },
        {
          id: 'amplitude_attack',
          display_name: 'Attack',
          min: 0,
          max: 5,
          scaling_factor: 1,
          default_value: 0.5,
          step: 0.01,
          scale_type: 'linear',
        },
        {
          id: 'amplitude_decay',
          display_name: 'Decay',
          min: 0,
          max: 5,
          scaling_factor: 1,
          default_value: 0.5,
          step: 0.01,
          scale_type: 'linear',
        },
        {
          id: 'amplitude_sustain',
          display_name: 'Sustain',
          min: 0,
          max: 1,
          scaling_factor: 1,
          default_value: 0.7,
          step: 0.01,
          scale_type: 'linear',
        },
        {
          id: 'amplitude_release',
          display_name: 'Release',
          min: 0,
          max: 5,
          scaling_factor: 1,
          default_value: 0.5,
          step: 0.01,
          scale_type: 'linear',
        },
      ];

</script>

<template>
  <main>
    <h2>Hi, test this will be the awesome synth interface</h2>

    <!-- Knob Controls -->
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

    <!-- Audio Player -->
    <div class="audio-player-section">
      <AudioPlayer audioSrc="/synth_saw_decay.wav" />
    </div>

    <!-- Line Chart Component -->
    <ADSRChart />

    <LoggingComponent />

      <div class="controls-section">
        <!-- Knob Controls Section -->
        <MatchKnobs :knobs="knobs" :knobValues="knobValues" />
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

.audio-player-section {
  margin-bottom: 1rem;
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

.adsr-plotter-section {
  width: 100%;
  max-width: 800px;
  margin-top: 2rem;
  text-align: center;
}
</style>
