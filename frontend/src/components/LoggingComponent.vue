<template>
  <!-- No visible content, as logs will be sent to the console -->
</template>

<script setup>
import { onMounted } from 'vue'
import { io } from 'socket.io-client'

// Initialize WebSocket connection
const initializeWebSocket = () => {
  const socket = io("http://127.0.0.1:5000");

  // Log WebSocket connection events
  socket.on("connect", () => {
    console.log('WebSocket connected');
  });

  socket.on("disconnect", () => {
    console.log('WebSocket disconnected');
  });

  socket.on("connect_error", (err) => {
    console.log(`WebSocket connection error: ${err}`);
  });

  // Listen for 'send_match' events from the backend
  socket.on("send_match", (data) => {
    console.log(`Received 'send_match' data: ${JSON.stringify(data)}`);
  });
};

// Call the WebSocket initialization when the component is mounted
onMounted(() => {
  initializeWebSocket();
});
</script>

<style scoped>
/* No need for styling since there's no UI to display */
</style>
