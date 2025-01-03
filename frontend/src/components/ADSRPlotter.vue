<template>
    <div class="adsr-plotter">
      <h4>Amplitude ADSR</h4>
      <canvas id="adsr-chart"></canvas>
    </div>
  </template>
  
  <script>
  import { onMounted } from 'vue';
  import { Chart, registerables } from 'chart.js';
  
  Chart.register(...registerables);
  
  export default {
    name: 'ADSRPlotter',
    props: {
      adsrData: {
        type: Array,
        required: true,
      },
    },
    setup(props) {
      const generateEnvelope = (attack, decay, sustainLevel, sustainLength, release, samplingRate = 100) => {
        const downsample = (data, step) => data.filter((_, index) => index % step === 0);
  
        const attackSamples = Math.floor(attack * samplingRate);
        const decaySamples = Math.floor(decay * samplingRate);
        const sustainSamples = Math.floor(sustainLength * samplingRate);
        const releaseSamples = Math.floor(release * samplingRate);
  
        const envelope = [
          ...Array.from({ length: attackSamples }, (_, i) => i / attackSamples),
          ...Array.from({ length: decaySamples }, (_, i) =>
            1 - ((1 - sustainLevel) * i) / decaySamples
          ),
          ...Array.from({ length: sustainSamples }, () => sustainLevel),
          ...Array.from({ length: releaseSamples }, (_, i) =>
            sustainLevel * (1 - i / releaseSamples)
          ),
          0,
        ];
  
        const time = [
          ...Array.from({ length: attackSamples + decaySamples + sustainSamples + releaseSamples },
            (_, i) => i / samplingRate
          ),
          (attackSamples + decaySamples + sustainSamples + releaseSamples) / samplingRate,
        ];
  
        const step = Math.max(1, Math.floor(envelope.length / 50));
        return {
          envelope: downsample(envelope, step),
          time: downsample(time, step),
        };
      };
  
      onMounted(() => {
        const rootStyle = getComputedStyle(document.documentElement);
  
        // Retrieve colors from global CSS variables
        const targetColor = rootStyle.getPropertyValue('--color-target').trim();
        const matchColor = rootStyle.getPropertyValue('--color-match').trim();
  
        const datasets = props.adsrData.map((adsr, index) => {
          const [attack, decay, sustainLevel, sustainLength, release] = adsr;
          const { envelope, time } = generateEnvelope(
            attack,
            decay,
            sustainLevel,
            sustainLength,
            release
          );
  
          return {
            label: `${index === 0 ? 'Target' : 'Match'}`,
            data: time.map((t, i) => ({ x: t, y: envelope[i] })),
            borderColor: index === 0 ? targetColor : matchColor,
            backgroundColor: index === 0 ? targetColor : matchColor, // Fill legend color
            borderWidth: 2,
            tension: 0,
            pointRadius: 0,
            pointHoverRadius: 0,
          };
        });
  
        new Chart(document.getElementById('adsr-chart'), {
          type: 'line',
          data: {
            datasets,
          },
          options: {
            plugins: {
              legend: {
                labels: {
                  boxWidth: 20, // Set box width for square shape
                  boxHeight: 10, // Explicitly set height for square boxes
                  usePointStyle: false, // Ensure square boxes
                },
              },
            },
            scales: {
              x: {
                type: 'linear',
                title: {
                  display: true,
                  text: 'Time (s)',
                },
              },
              y: {
                title: {
                  display: true,
                  text: 'Amplitude',
                },
                min: 0,
                max: 1.2,
              },
            },
          },
        });
      });
    },
  };
  </script>
  
  <style scoped>
  .adsr-plotter {
    margin: 2rem auto;
    width: 100%;
    max-width: 800px;
  }
  canvas {
    width: 100%;
    height: 400px;
  }
  </style>
  