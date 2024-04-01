from src.synthesizers.base_synth import Synthesizer
from dataclasses import dataclass
import numpy as np


class SuperSimpleSynth(Synthesizer):
    def __init__(self, sample_rate=48000.):
        super().__init__(sample_rate)
        self.SP = SynthParameters(
            amp=2 * np.sqrt(2),
            mod_freq=2,
            mod_amp=500
        )

    def play_note(self, note, duration):
        # Convert note to frequency
        freq = _note_to_freq(note)

        # Generate array of time points
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # Generate sound wave
        modulator = self.SP.mod_amp * np.sin(2 * np.pi * self.SP.mod_freq * t)
        sound = self.SP.amp * np.sin(2 * np.pi * freq * t + modulator)

        return sound

    def get_parameters(self):
        return np.array([self.SP.amp, self.SP.mod_freq, self.SP.mod_amp])

    def set_parameters(self, parameters):
        self.SP.amp = parameters[0]
        self.SP.mod_freq = parameters[1]
        self.SP.mod_amp = parameters[2]


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


@dataclass
class SynthParameters:
    amp: float
    mod_freq: float
    mod_amp: float
