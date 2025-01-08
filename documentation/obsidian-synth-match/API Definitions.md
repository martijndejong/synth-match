### Send target parameters & generate target sound
- **Input**
	- Synth parameters
	- Note length (disabled for now)
	- Note (C4 etc - disabled for now)
	- (Later we would like to specify n_mels & hop_len, but out of scope for now)
- **Return**
	- Target Sound (virtualized?) - same resolution as in Environment
	- Target Spectrogram - same resolution as in Environment, maybe bit reduced
	- ~~(Target ADSR point values (depends if the logic is centralized in the BE or not))~~
	- **Potential format**:
		- {
			- target_sound: {
					- Float array [x * 1] - TBD if and how we will handle stereo
				- },
			- target_spectogram: {
				- - Float array [m x n] - TBD - format can later change depending on n_mels & hop_len
			- }
		- }

**Example request**
```json
{
    "note": "C4",
    "note_length": 0.5,
    "target_parameters": {
      "amplitude": 0.8,
      "amplitude_attack": 0.5,
      "amplitude_decay": 0.3,
      "amplitude_sustain": 0.2,
      "amplitude_release": 0.9,
      "filter_cutoff_frequency": 0.3,
      "filter_cutoff_resonance": 1.2
    }
  }
```
**Example return**
```json
{

    "target_audio_data": {
      "sample_rate": 44100,
      "bit_depth": 16,
      "samples": [0, 324, 643, -234, -576, -304, 384, 203, -203]
    },
    "target_spectrogram_data": {
      "width": 256,
      "height": 128,
      "pixel_values": [
        [0, 32, 15, 12],
        [12, 45, 78, 16],
        [18, 62, 43, 24],
        [41, 45, 64, 44]
      ]
    }
}
```

### Get match from parameters
**Example request**
```json
{
    "note": "C4", // Hardcoded in the beginning, configurable later
    "note_length": 0.5, // Hardcoded in the beginning, configurable later
    "target_parameters": {
      "amplitude": 0.8,
      "amplitude_attack": 0.8,
      "amplitude_decay": 0.8,
      "amplitude_sustain": 0.8,
      "amplitude_release": 0.8,
      "filter_cutoff_frequency": 0.3,
      "filter_cutoff_resonance": 1.2
    }
  }
```
**Example return**
```json
{
	"matched_parameters": {
      "amplitude": 0.8,
      "amplitude_attack": 0.8,
      "amplitude_decay": 0.8,
      "amplitude_sustain": 0.8,
      "amplitude_release": 0.8,
      "filter_cutoff_frequency": 0.3,
      "filter_cutoff_resonance": 1.2
    },
    "matched_audio_data": {
      "sample_rate": 44100,
      "bit_depth": 16,
      "samples": [0, 324, 643, -234, -576, -304, 384, 203, -203]
    },
    "matched_spectrogram_data": {
      "width": 256,
      "height": 128,
      "pixel_values": [
        [0, 32, 15, 12],
        [12, 45, 78, 16],
        [18, 62, 43, 24],
        [41, 45, 64, 44]
      ]
    }
}```


