from src.synthesizers._base_synth import BaseSynthesizer
from src.utils.math import linear_interp
import numpy as np


class SuperSimpleSynth(BaseSynthesizer):
    def __init__(self, sample_rate=48000.):
        super().__init__(sample_rate)
        # self.SP = SynthParameters(
        #     amp=2 * np.sqrt(2),
        #     mod_freq=2,
        #     mod_amp=500
        # )

        self.param_names = [
            # "amp",
            "mod_freq",
            "mod_amp"
        ]

        self.param_values = [
            # 0.5,
            0.5,
            0.5
        ]

        self.param_range = [
            # (0.0, 5 * np.sqrt(2)),
            (0.0, 8.0),
            (0.0, 1000.0)
        ]

    def play_note(self, note, duration):
        # Convert note to frequency
        if type(note) == str:
            freq = BaseSynthesizer._note_to_freq(note)
        elif type(note) == int:
            freq = BaseSynthesizer._note_number_to_freq(note)
        else:
            TypeError("Note should be either string (e.g., 'C4'), or int (e.g., 60)")

        # FIXME: not the nicest way to explicitly update each parameter here each time
        #   but this synth is for testing only anyway
        # amp_range = self.param_range[0]
        # amp = linear_interp(amp_range[0], amp_range[1], self.param_values[0])

        mod_freq_range = self.param_range[0]
        mod_freq = linear_interp(mod_freq_range[0], mod_freq_range[1], self.param_values[0])

        mod_amp_range = self.param_range[1]
        mod_amp = linear_interp(mod_amp_range[0], mod_amp_range[1], self.param_values[1])

        # Generate array of time points
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # FIXME: hardcoded amp value to ignore it as tunable parameter
        amp = 2 * np.sqrt(2)

        # Generate sound wave
        modulator = mod_amp * np.sin(2 * np.pi * mod_freq * t)
        sound = amp * np.sin(2 * np.pi * freq * t + modulator)

        return sound

    def get_param_value(self, index: int) -> float:
        return self.param_values[index]

    def get_param_name(self, index: int) -> str:
        return self.param_names[index]

    def set_param_value(self, index: int, value: float) -> None:
        self.param_values[index] = max(0.0, min(1.0, value))  # value bounds [0.0, 1.0]
        return

    @property
    def num_params(self):
        return len(self.param_values)
