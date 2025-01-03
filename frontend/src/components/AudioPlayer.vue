// Install necessary packages
// npm install vue primevue primeicons

<template>
  <div class="audio-player">
    <div class="controls">
      <Button 
        :icon="isPlaying ? 'pi pi-pause' : 'pi pi-play'"
        @click="togglePlayPause"
        class="p-button-rounded p-button-text" />
      <Slider 
        v-model="sliderValue" 
        :max="duration" 
        :step="1" 
        class="slider" 
        @change="onSliderChange" />
      <span class="timestamp">{{ formatTime(currentTime) }} / {{ formatTime(duration) }}</span>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue';
import Button from 'primevue/button';
import Slider from 'primevue/slider';

export default {
  name: 'AudioPlayer',
  components: { Button, Slider },
  props: {
    audioSrc: {
      type: String,
      required: true
    }
  },
  setup(props) {
    const audio = ref(null);
    const isPlaying = ref(false);
    const currentTime = ref(0);
    const duration = ref(0);
    const sliderValue = ref(0);

    const togglePlayPause = () => {
      if (audio.value) {
        if (isPlaying.value) {
          audio.value.pause();
        } else {
          audio.value.play();
        }
        isPlaying.value = !isPlaying.value;
      }
    };

    const formatTime = (seconds) => {
      const min = Math.floor(seconds / 60);
      const sec = Math.floor(seconds % 60).toString().padStart(2, '0');
      return `${min}:${sec}`;
    };

    const onSliderChange = (e) => {
      if (audio.value) {
        audio.value.currentTime = e.value;
      }
    };

    onMounted(() => {
      audio.value = new Audio(props.audioSrc);
      audio.value.addEventListener('loadedmetadata', () => {
        duration.value = audio.value.duration;
      });

      audio.value.addEventListener('timeupdate', () => {
        currentTime.value = audio.value.currentTime;
        sliderValue.value = audio.value.currentTime;
      });

      audio.value.addEventListener('ended', () => {
        isPlaying.value = false; // Reset to play button when audio ends
        currentTime.value = 0; // Reset timestamp
        sliderValue.value = 0; // Reset slider
      });
    });

    watch(currentTime, (newTime) => {
      sliderValue.value = newTime;
    });

    return {
      isPlaying,
      currentTime,
      duration,
      sliderValue,
      togglePlayPause,
      formatTime
    };
  }
};
</script>

<style scoped>
.audio-player {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1rem;
  border-radius: 50px;
  background-color: #ffffff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  width: fit-content;
  margin: auto;
}

.controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.timestamp {
  font-size: 0.9rem;
  color: #555;
}

.slider {
  width: 150px;
}
</style>
