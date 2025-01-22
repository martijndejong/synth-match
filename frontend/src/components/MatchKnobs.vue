<!-- TODO - Add scaling of match synth parameters to min max range of parameters and linear/logarithmic scaling  -->
<template>
  <div class="button-row">
    <div v-for="knob in localKnobs" :key="knob.id" class="knob-item">
      <!-- Knob component for controlling values -->
      <Knob
        v-model="knob.default_value"
        :min="knob.min"
        :max="knob.max"
        :step="knob.step"
        valueColor="#4287f5"
        readonly 
      />
      <span class="knob-label">{{ knob.display_name }}</span>
    </div>
  </div>
</template>

<script>
import { reactive, watch } from 'vue';
import Knob from 'primevue/knob';
import { io } from 'socket.io-client';

export default {
  name: 'MatchKnobs',
  components: {
    Knob,
  },
  props: {
    knobs: {
      type: Array,
      required: true,
    },
  },
  data() {
    return {
      localKnobs: reactive([...this.knobs]), // Create a local reactive copy
      socket: null, // Socket connection
    };
  },
  methods: {
    initSocket() {
      this.socket = io('http://127.0.0.1:5000'); // Replace with your socket server URL

      this.socket.on('send_match', (data) => {
        if (data.matched_parameters) {
          console.log('Updating knobs with matched parameters');
          this.updateKnobValues(data.matched_parameters);
        }
      });

      this.socket.on('connect', () => console.log('Socket connected'));
      this.socket.on('disconnect', () => console.log('Socket disconnected'));
    },
    setKnobValue(id, value) {
      const knob = this.localKnobs.find((k) => k.id === id);
      console.log("trying to find", id);
      console.log("in: ");
      console.log('Knob IDs:', this.localKnobs.map(knob => knob.id));
      if (knob) {
        knob.default_value = value;        
      }
    },
    updateKnobValues(matchedParameters) {
      for (const [id, value] of Object.entries(matchedParameters)) {
        const rounded_value = value.toFixed(2)
        console.log('Setting knob:', id, 'to value:', rounded_value)
        this.setKnobValue(id, rounded_value);
      }
    },
  },
  mounted() {
    this.initSocket();
  },
  beforeUnmount() {
    if (this.socket) {
      this.socket.disconnect();
    }
  },
};
</script>

<style>
.knob-container {
  margin-bottom: 20px;
}
.knob-label {
  font-size: 1rem;
  margin-top: 0.5rem;
  font-weight: bold;
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
</style>
