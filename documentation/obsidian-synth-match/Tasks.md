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
- [x] Central configuration file in backend for parameters
	- ~~With parameters e.g.: name, is_visualized, min_val, max_val, color ~~
- [x] Add dynamic matched parameters to visualize match synth settings
- [x] Integrate ADSR plot with matched parameter API call
- [x] Add scaling to match knobs to match min/max and linear/logarithmic
#### Ideas
##### ML
- [ ] Increase parameter space with cutoff frequency ADSR
- [ ] Increase parameter space with second oscillator
- [ ] Increase parameter space with oscillator 1 vs oscillator 2 mix
- [ ] Create option to dynamically change the environment reward function (Euclidean -> Image similarity)
##### Frontend
- [ ] Method to visualize spectrogram in other resolution frequency than used in audio processor internal state
- [ ] Draft initial training validation UI design
- [ ] Add button to generate sound based on synth preset
	- [ ] Optionally make the sound bar grey when no sound is selected yet
- [ ] Create time plots comparison functionality
- [ ] Create spectrogram functionality
- [ ] Create match audioplayer and integrate with match API call
- [ ] Make ADSR plot time settings dynamic from the match parameter output
- [ ] Centralize ADSR plot styling
- [ ] Improve target synth knobs (see TODOs in file)

##### Integration
- [x] Draft API design