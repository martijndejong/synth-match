import './assets/main.css'

import { createApp } from 'vue';
import App from './App.vue';
import PrimeVue from 'primevue/config';
import Aura from '@primevue/themes/aura';
// removed the following due to Primevue update not having resources anymore
// import 'primevue/resources/themes/saga-blue/theme.css'; // Choose your theme
// import 'primevue/resources/primevue.min.css';
// import 'primeicons/primeicons.css';
// import './assets/main.css';

const app = createApp(App);

app.use(PrimeVue, {
    // Default theme configuration
    theme: {
        preset: Aura,
        options: {
            prefix: 'p',
            darkModeSelector: 'system',
            cssLayer: false
        }
    }
 });

app.mount('#app');