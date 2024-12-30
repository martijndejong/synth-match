<template>
    <div>
      <Knob v-model="knobValue" @change="updateBackend" :min="0" :max="100" />
    </div>
  </template>
  
  <script>
  import Knob from "primevue/knob";
  import axios from "axios";
  import { io } from "socket.io-client";
  
  export default {
    components: { Knob },
    props: {
      id: {
        type: String,
        required: true, // Ensures each knob receives an ID
      },
    },
    data() {
      return {
        knobValue: 50, // Default knob value
        socket: null,  // WebSocket connection
      };
    },
    methods: {
      // Fetch initial knob value from backend
      async fetchKnobValue() {
        try {
          const response = await axios.get(`http://127.0.0.1:5000/get-knob`, {
            params: { id: this.id }, // Send the unique knob ID
          });
          this.knobValue = response.data.value; // Update knob value with backend response
          console.log(`Fetched value for ${this.id}:`, this.knobValue);
        } catch (error) {
          console.error(`Error fetching value for ${this.id}:`, error);
        }
      },
  
      // Send updated knob value to backend
      async updateBackend() {
        try {
          const response = await axios.post("http://127.0.0.1:5000/update-knob", {
            id: this.id, // Include the unique identifier
            value: this.knobValue,
          });
          console.log("Backend response:", response.data);
        } catch (error) {
          console.error("Error updating backend:", error);
        }
      },
  
      // Initialize WebSocket connection
      initializeWebSocket() {
        this.socket = io("http://127.0.0.1:5000"); // Connect to the backend WebSocket
        this.socket.on("knob_update", (data) => {
          if (data.id === this.id) {
            this.knobValue = data.value; // Update knob value if the ID matches
            console.log(`Knob ${data.id} updated to ${data.value}`);
          }
        });
      },
    },
    mounted() {
      // Fetch knob value when component is mounted
      this.fetchKnobValue();
      // Initialize WebSocket connection
      this.initializeWebSocket();
    },
    beforeDestroy() {
      // Disconnect WebSocket on component destruction
      if (this.socket) {
        this.socket.disconnect();
      }
    },
  };
  </script>
  
  <style scoped>
  div {
    text-align: center;
    margin-top: 50px;
  }
  </style>
  