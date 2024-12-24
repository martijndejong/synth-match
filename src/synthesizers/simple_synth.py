from src.synthesizers._base_synth import BaseSynthesizer
from src.utils.math import linear_interp
import numpy as np
from scipy.signal import butter, lfilter

# Import waveforms
from src.synthesizers.waveforms import generate_sine, generate_saw


class SimpleSynth(BaseSynthesizer):
    def __init__(self, sample_rate=48000.):
        super().__init__(sample_rate)

        self.param_names = [
            # "amp",
            "attack",
            "decay",
            "sustain",
            "release",
            "cutoff_freq"
        ]

        self.param_values = [
            # 0.5,  # amp
            0.1,  # attack
            0.1,  # decay
            0.7,  # sustain
            0.1,  # release
            0.5   # cutoff_freq
        ]

        self.param_range = [
            # (0.0, 2 * np.sqrt(2)),  # amp
            (0.0, 1.0),  # attack time in seconds
            (0.0, 0.5),  # decay time in seconds
            (0.0, 1.0),  # sustain level
            (0.0, 1.0),  # release time in seconds
            (20.0, 10000.0)  # cutoff freq in Hz
        ]

        # Set default waveform generator here (e.g. saw)
        self.waveform_generator = generate_saw
        # To use a sine wave instead, you could do:
        # self.waveform_generator = generate_sine

    def play_note(self, note, duration):
        # Convert note to frequency
        if isinstance(note, str):
            freq = BaseSynthesizer._note_to_freq(note)
        elif isinstance(note, int):
            freq = BaseSynthesizer._note_number_to_freq(note)
        else:
            raise TypeError("Note should be either string (e.g., 'C4'), or int (e.g., 60)")

        # Retrieve and map parameters
        amp = 0.01  # linear_interp(self.param_range[0][0], self.param_range[0][1], self.param_values[0])
        attack = linear_interp(self.param_range[0][0], self.param_range[0][1], self.param_values[0])
        decay = linear_interp(self.param_range[1][0], self.param_range[1][1], self.param_values[1])
        sustain = linear_interp(self.param_range[2][0], self.param_range[2][1], self.param_values[2])
        release = linear_interp(self.param_range[3][0], self.param_range[3][1], self.param_values[3])
        cutoff_freq = linear_interp(self.param_range[4][0], self.param_range[4][1], self.param_values[4])

        sample_rate = self.sample_rate

        # Compute number of samples for each stage
        attack_samples = int(attack * sample_rate)
        decay_samples = int(decay * sample_rate)
        release_samples = int(release * sample_rate)
        total_note_on_samples = int(duration * sample_rate)

        # Total samples for attack and decay phases
        attack_decay_samples = attack_samples + decay_samples

        # Initialize envelope arrays
        envelope = np.array([])

        # Handle different scenarios based on note duration
        if total_note_on_samples <= attack_samples:
            # Released during attack
            attack_envelope = np.linspace(0, 1, attack_samples, endpoint=False)[:total_note_on_samples]
            current_amplitude = attack_envelope[-1] if len(attack_envelope) > 0 else 1
            release_envelope = np.linspace(current_amplitude, 0, release_samples, endpoint=False)
            envelope = np.concatenate((attack_envelope, release_envelope))
        elif total_note_on_samples <= attack_decay_samples:
            # Released during decay
            attack_envelope = np.linspace(0, 1, attack_samples, endpoint=False)
            decay_duration_samples = total_note_on_samples - attack_samples
            decay_envelope = np.linspace(1, sustain, decay_samples, endpoint=False)[:decay_duration_samples]
            envelope = np.concatenate((attack_envelope, decay_envelope))
            current_amplitude = envelope[-1] if len(envelope) > 0 else sustain
            release_envelope = np.linspace(current_amplitude, 0, release_samples, endpoint=False)
            envelope = np.concatenate((envelope, release_envelope))
        else:
            # Sustain phase reached
            attack_envelope = np.linspace(0, 1, attack_samples, endpoint=False)
            decay_envelope = np.linspace(1, sustain, decay_samples, endpoint=False)
            sustain_duration_samples = total_note_on_samples - attack_samples - decay_samples
            sustain_envelope = np.full(sustain_duration_samples, sustain)
            envelope = np.concatenate((attack_envelope, decay_envelope, sustain_envelope))
            release_envelope = np.linspace(sustain, 0, release_samples, endpoint=False)
            envelope = np.concatenate((envelope, release_envelope))

        # Generate time array
        total_samples = len(envelope)
        t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)

        # Generate waveform using the selected waveform generator
        waveform = self.waveform_generator(freq, t, amp)

        # Apply envelope
        sound = waveform * envelope

        # Apply low-pass filter
        sound = lowpass_filter(sound, cutoff_freq, self.sample_rate)

        return sound

    def get_param_value(self, index: int) -> float:
        return self.param_values[index]

    def get_param_name(self, index: int) -> str:
        return self.param_names[index]

    def set_param_value(self, index: int, value: float) -> None:
        # Ensure value is within [0.0, 1.0]
        self.param_values[index] = max(0.0, min(1.0, value))

    @property
    def num_params(self):
        return len(self.param_values)


def lowpass_filter(data, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data
