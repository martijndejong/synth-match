from src.synthesizers._base_synth import BaseSynthesizer
from src.utils.math import linear_interp
import numpy as np

# Import simple synth utilities
from src.synthesizers.simple_synth_utils import generate_sine, generate_saw, generate_adsr, variable_lowpass_filter


class SimpleSynth(BaseSynthesizer):
    def __init__(self, sample_rate=48000.):
        super().__init__(sample_rate)

        # Define separate ADSR parameters for amplitude and for cutoff.
        self.param_names = [
            "amp_attack",
            "amp_decay",
            "amp_sustain",
            "amp_release",
            "cutoff_attack",
            "cutoff_decay",
            "cutoff_sustain",
            "cutoff_release",
            "cutoff_freq"
        ]
        self.param_values = [
            0.1,  # amp_attack (seconds)
            0.1,  # amp_decay (seconds)
            0.7,  # amp_sustain (level, 0–1)
            0.1,  # amp_release (seconds)
            0.05,  # cutoff_attack (seconds)
            0.1,  # cutoff_decay (seconds)
            0.7,  # cutoff_sustain (level, 0–1)
            0.1,  # cutoff_release (seconds)
            0.5  # cutoff_freq (normalized value, mapped to Hz)
        ]
        self.param_range = [
            (0.0, 1.0),  # amp_attack in sec
            (0.0, 0.5),  # amp_decay in sec
            (0.0, 1.0),  # amp_sustain (level)
            (0.0, 1.0),  # amp_release in sec
            (0.0, 1.0),  # cutoff_attack in sec
            (0.0, 0.5),  # cutoff_decay in sec
            (0.0, 1.0),  # cutoff_sustain (level)
            (0.0, 1.0),  # cutoff_release in sec
            (20.0, 10000.0)  # cutoff_freq in Hz
        ]

        # Select a default waveform generator.
        self.waveform_generator = generate_saw
        # For a sine wave, you could do:
        # self.waveform_generator = generate_sine

    def play_note(self, note, duration):
        # Convert note (e.g., "C4" or MIDI number) to frequency.
        if isinstance(note, str):
            freq = BaseSynthesizer._note_to_freq(note)
        elif isinstance(note, int):
            freq = BaseSynthesizer._note_number_to_freq(note)
        else:
            raise TypeError("Note must be a string (e.g., 'C4') or an integer (e.g., 60)")

        sample_rate = self.sample_rate

        # Retrieve amplitude ADSR parameters.
        amp_attack = linear_interp(self.param_range[0][0], self.param_range[0][1], self.param_values[0])
        amp_decay = linear_interp(self.param_range[1][0], self.param_range[1][1], self.param_values[1])
        amp_sustain = linear_interp(self.param_range[2][0], self.param_range[2][1], self.param_values[2])
        amp_release = linear_interp(self.param_range[3][0], self.param_range[3][1], self.param_values[3])

        # Retrieve cutoff ADSR parameters.
        cutoff_attack = linear_interp(self.param_range[4][0], self.param_range[4][1], self.param_values[4])
        cutoff_decay = linear_interp(self.param_range[5][0], self.param_range[5][1], self.param_values[5])
        cutoff_sustain = linear_interp(self.param_range[6][0], self.param_range[6][1], self.param_values[6])
        cutoff_release = linear_interp(self.param_range[7][0], self.param_range[7][1], self.param_values[7])
        cutoff_freq = linear_interp(self.param_range[8][0], self.param_range[8][1], self.param_values[8])

        # Generate the two envelopes.
        amp_env = generate_adsr(amp_attack, amp_decay, amp_sustain, amp_release, duration, sample_rate)
        cutoff_env = generate_adsr(cutoff_attack, cutoff_decay, cutoff_sustain, cutoff_release, duration, sample_rate)
        # Multiply the unit–valued cutoff envelope by the target cutoff frequency.
        cutoff_env = cutoff_env * cutoff_freq

        # To ensure both envelopes (and thus the final note length) match,
        # we pad the shorter one with its last value.
        total_samples = max(len(amp_env), len(cutoff_env))
        if len(amp_env) < total_samples:
            amp_env = np.pad(amp_env, (0, total_samples - len(amp_env)), mode='edge')
        if len(cutoff_env) < total_samples:
            cutoff_env = np.pad(cutoff_env, (0, total_samples - len(cutoff_env)), mode='edge')

        # Generate the common time array.
        t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)

        # Use a fixed amplitude for waveform generation (the envelope shapes the loudness).
        amp = 0.01
        waveform = self.waveform_generator(freq, t, amp)

        # Apply the amplitude envelope.
        sound = waveform * amp_env

        # Pass the sound through the variable low–pass filter using the cutoff envelope.
        filtered_sound = variable_lowpass_filter(sound, cutoff_env, sample_rate)

        return filtered_sound

    def get_param_value(self, index: int) -> float:
        return self.param_values[index]

    def get_param_name(self, index: int) -> str:
        return self.param_names[index]

    def set_param_value(self, index: int, value: float) -> None:
        # Clamp the value to the range [0.0, 1.0].
        self.param_values[index] = max(0.0, min(1.0, value))

    @property
    def num_params(self):
        return len(self.param_values)
