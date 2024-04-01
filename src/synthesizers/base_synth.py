"""
Abstract base class for synthesizers, defining a set of functions each synthesizer must contain
"""

from abc import ABC, abstractmethod

import numpy as np


class Synthesizer(ABC):
    def __init__(self, sample_rate=48000.):
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
    def get_parameters(self) -> np.ndarray:
        """
        Get the current synthesizer parameters.
        :return: numpy array with synthesizer parameters
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: np.ndarray) -> None:
        """
        Set the synthesizer parameters.
        :param parameters: input numpy array with synthesizer settings.
        :return: None
        """
        pass
