from src.synthesizers._base_synth import Synthesizer
from src.utils.math import linear_interp
import numpy as np
# from dataclasses import dataclass, fields

class SuperSimpleHost():
    def __init__(self, sample_rate=44100.) -> None:
        self.vst = SuperSimpleSynth(sample_rate=sample_rate)

    def play_note(self, note, note_duration):
        return self.vst.play_note(note=note, duration=note_duration)

class SuperSimpleSynth(Synthesizer):
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
            (0.0, 10.0),
            (0.0, 1000.0)
        ]

    def play_note(self, note, duration):
        # Convert note to frequency
        if type(note) == str:
            freq = _note_to_freq(note)
        elif type(note) == int:
            freq = _note_number_to_freq(note) 
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


def _note_to_freq(note):
    # Musical notes with their corresponding number of half steps from A4
    notes = {
        'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6, 'E': -5,
        'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1, 'Ab': -1, 'A': 0,
        'A#': 1, 'Bb': 1, 'B': 2
    }

    # Extract the note and the octave from the input
    note_letter = note[:-1]
    octave = int(note[-1])

    if note_letter not in notes:
        raise ValueError("Invalid note name")

    # Calculate the number of half steps from A4
    n = notes[note_letter] + (octave - 4) * 12

    # Calculate the frequency
    frequency = 2 ** (n / 12) * 440
    return frequency

def _note_number_to_freq(note_number):
    """
    Convert a MIDI note number to its corresponding frequency in Hz.
    With 69 corresponding to A4 (440 Hz)
    
    Parameters:
        note_number (int): The MIDI note number.
        
    Returns:
        float: The frequency of the note in Hz.
    """
    return 440.0 * (2 ** ((note_number - 69) / 12.0))


# @dataclass
# class SynthParameters:
#     amp: float
#     mod_freq: float
#     mod_amp: float
