<template>
    <div id="scatterchart" class="chart-container">
      <Chart type="scatter" :data="chartData" :options="chartOptions" style="width: 100%; height: 500px;" />
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import Chart from 'primevue/chart';
  import { io } from 'socket.io-client';
  
  // Reactive object to store the current values for Attack, Decay, Sustain, Release
  const adsrValues = ref({
    Attack: 100,  // Default value
    Decay: 100,   // Default value
    Sustain: 100, // Default value
    Release: 100, // Default value
  });

  // TODO: Make these dynamic from the match parameter output
  const match_adsr_values = ref({
    Attack: 30,  // Constant value
    Decay: 50,   // Constant value
    Sustain: 30, // Constant value
    Release: 70, // Constant value
  });
  
  // Reactive array to store chart data
  const chartData = ref({
    labels: [], // Empty initially, we will set it dynamically
    datasets: [
      {
        label: 'Target Amplitude Envelope', // ADSR curve influenced by API calls
        data: [], // Dynamic envelope data will be filled here
        borderColor: '#42b883',  // Line color
        pointBackgroundColor: '#42b883', // Scatter points color
        tension: 0.0, // No curve smoothing
        fill: false,
        showLine: true, // Show line connecting the points
        pointRadius: 3, // Scatter point size
      },
      {
        label: 'Match Amplitude Envelope', // Hardcoded ADSR curve
        data: [], // Fixed envelope data will be filled here
        borderColor: '#4277b8',  // Line color for the fixed ADSR
        pointBackgroundColor: '#4277b8', // Scatter points color
        tension: 0.0, // No curve smoothing
        fill: false,
        showLine: true, // Show line connecting the points
        pointRadius: 3, // Scatter point size
      },
    ],
  });
  
  // Chart options to disable animations and configure the X-axis
  const chartOptions = ref({
    responsive: true,
    maintainAspectRatio: false,
    animation: false, // Disable animations
    plugins: {
      legend: {
        labels: {
                  boxWidth: 20, // Set box width for square shape
                  boxHeight: 10, // Explicitly set height for square boxes
                  usePointStyle: false, // Ensure square boxes

                //   fillStyle: Color,
                },
      },
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'Time (seconds)',
        },
        ticks: {
          autoSkip: false,
          precision: 2,
          callback: (value) => value.toFixed(2),
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
  });
  
  // Helper function to generate the dynamic ADSR envelope based on stored values
  function generateDynamicADSREnvelope() {
    const { Attack, Decay, Sustain, Release } = adsrValues.value;
  
    // Time durations in seconds, based on knob values (0.5s max)
    const attackTime = (Attack / 100) * 0.5;
    const decayTime = (Decay / 100) * 0.5;
    const sustainLevel = Sustain / 100;
    const releaseTime = (Release / 100) * 0.5;
  
    // Time points for dynamic envelope
    const timePointsDynamic = [
      0,
      attackTime,
      attackTime + decayTime,
      attackTime + decayTime + 0.5,
      attackTime + decayTime + 0.5 + releaseTime,
    ];
  
    const amplitudeValuesDynamic = [
      0,
      1,
      sustainLevel,
      sustainLevel,
      0,
    ];
  
    // Update the chart data for the dynamic envelope
    chartData.value.datasets[0].data = timePointsDynamic.map((time, index) => ({
      x: time,  // x-axis value (time)
      y: amplitudeValuesDynamic[index],  // y-axis value (amplitude)
    }));
    chartData.value.labels = timePointsDynamic.map((time) => time.toFixed(2)); // Set labels for dynamic envelope
  }
  
  // Helper function to generate the fixed ADSR envelope based on hardcoded values
  function generateFixedADSREnvelope() {
    const { Attack, Decay, Sustain, Release } = match_adsr_values.value;
  
    // Time durations in seconds, based on knob values (0.5s max)
    const attackTime = (Attack / 100) * 0.5;
    const decayTime = (Decay / 100) * 0.5;
    const sustainLevel = Sustain / 100;
    const releaseTime = (Release / 100) * 0.5;
  
    // Time points for fixed envelope
    const timePointsFixed = [
      0,
      attackTime,
      attackTime + decayTime,
      attackTime + decayTime + 0.5,
      attackTime + decayTime + 0.5 + releaseTime,
    ];
  
    const amplitudeValuesFixed = [
      0,
      1,
      sustainLevel,
      sustainLevel,
      0,
    ];
  
    // Update the chart data for the fixed envelope
    chartData.value.datasets[1].data = timePointsFixed.map((time, index) => ({
      x: time,  // x-axis value (time)
      y: amplitudeValuesFixed[index],  // y-axis value (amplitude)
    }));
  }
  
  // Initialize WebSocket connection
  const socket = io('http://127.0.0.1:5000');
  
  // Function to update the specific ADSR value based on incoming knob updates
  socket.on('knob_update', (data) => {
    if (data.id in adsrValues.value) {
      adsrValues.value[data.id] = data.value;
      generateDynamicADSREnvelope(); // Regenerate the dynamic envelope
      generateFixedADSREnvelope();   // Regenerate the fixed envelope
    }
  
    chartData.value = { ...chartData.value }; // Update chart data
  });
  
  // Lifecycle hook
  onMounted(() => {
    console.log('ADSR Scatter Plot component has been mounted!');
    generateDynamicADSREnvelope(); // Generate the initial dynamic ADSR envelope
    generateFixedADSREnvelope();   // Generate the initial fixed ADSR envelope
  });
  </script>
  
  <style scoped>
  .chart-container {
    width: 100%;
    max-width: 800px;
    margin-top: 2rem;
    text-align: center;
  }
  </style>
  