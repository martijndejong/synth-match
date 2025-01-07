**Implemented**
- [x] Initial frontend setup
- [x] Initial backend setup
- [x] Implement two-way API connection for gauges
- [x] Create Audio processor component for frontend
- [x] Create ADSR plotter component for frontend
- [x] Fix observer training script inefficiency (load more samples per get_data() call)
- [x] Refactor observer training script to be config driven
- [x] Refactor observer training script to use wandb
- [x] Train observer with higher feature_dim (input size for agent)
- [x] Train observer with more n_mels spectrogram
- [x] Refactor 'scripts/train_agent.py' to resemble end_to_end training, just with frozen observer

#### Ideas
##### ML
- [ ] Increase parameter space with cutoff frequency ADSR
- [ ] Increase parameter space with second oscillator
- [ ] Increase parameter space with oscillator 1 vs oscillator 2 mix
- [ ] Create option to dynamically change the environment reward function (Euclidean -> Image similarity)
##### Frontend
- [ ] Central configuration file in backend for parameters
	- With parameters e.g.: name, is_visualized, min_val, max_val, color
- [ ] Method to visualize spectrogram in other resolution frequency than used in audio processor internal state
- [ ] Draft initial training validation UI design
- [ ] Add button to generate sound based on synth preset
	- [ ] Optionally make the sound bar grey when no sound is selected yet
- [ ] Create time plots comparison functionality
- [ ] Create spectrogram functionality 
##### Integration
- [ ] Draft API design