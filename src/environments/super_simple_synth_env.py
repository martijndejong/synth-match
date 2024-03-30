from src.environments.base_synth_env import SynthEnv
from dataclasses import dataclass
import numpy as np


class SuperSimpleSynth(SynthEnv):
    def __init__(self, render_mode=None, sample_rate=48000.):
        super().__init__(render_mode, sample_rate)
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

    # TODO: ABSTRACT METHOD FOR SET PARAMETERS IN BASE CLASS?
    # TODO: 'CONTROL MODE' IN BASE CLASS -- CAN THIS ALSO SOMEHOW BE WRAPPED IN STEP?
    #   for example: base step function does pretty much what is shown below, just get/set/play are child overwritten
    # TODO: AND DERIVATIVE VS ABSOLUTE IN SET_PARAMETERS?
    def set_parameters(self, parameters):
        self.SP.amp = parameters[0]
        self.SP.mod_freq = parameters[1]
        self.SP.mod_amp = parameters[2]

    def step(self, action):
        current_params = self.get_parameters()

        if self.control_mode == 'incremental':
            new_params = current_params + action
            self.set_parameters(new_params)

        else:
            self.set_parameters(action)

        state = self.play_note(note='C4', duration=2.0)
        reward = 0
        done = False

        return state, reward, done

    def reset(self):
        # TODO: ADD ACTUAL RANGES FOR PARAMETERS SOMEWHERE -- THESE ARE ALSO IMPORTANT FOR AGENT
        random_amp = np.random.uniform(0.5 * np.sqrt(2), 2.5 * np.sqrt(2))
        random_mod_freq = np.random.uniform(0.0, 20.0)
        random_mod_amp = np.random.uniform(0.0, 1000)

        self.set_parameters([random_amp, random_mod_freq, random_mod_amp])
        state = self.play_note(note='C4', duration=2.0)

        return state


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
