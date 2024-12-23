"""
Abstract base class for synthesizers, defining a set of functions each synthesizer must contain
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseSynthesizer(ABC):
    def __init__(self, sample_rate=44100.):
        self.sample_rate = sample_rate
        pass  # Initialize shared attributes or configurations

    @abstractmethod
    def play_note(self, note: str, duration: float) -> np.ndarray:
        """
        Let synthesizer play an input note for given duration.
        :param note: input note (e.g. 'C4')
        :param duration: duration that input note is played
        :return: numpy array with the sound data
        """
        pass

    @abstractmethod
    def get_param_value(self, index: int) -> float:
        """
        Get the current synthesizer parameter value at index.
        :return: float value of parameter
        """
        pass

    @abstractmethod
    def get_param_name(self, index: int) -> str:
        """
        Get the current synthesizer parameter name at index.
        :return: string name of parameter
        """
        pass

    @abstractmethod
    def set_param_value(self, index: int, value: float) -> None:
        """
        Set synthesizer parameter at specified index.
        :param index: integer index of parameter to set
        :param value: float value to set parameter to
        :return: None
        """
        pass

    @property
    @abstractmethod
    def num_params(self):
        """
        Get the number of parameters the synthesizer has
        :return: integer value representing number of parameters
        """
        pass

    @staticmethod
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

    @staticmethod
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
