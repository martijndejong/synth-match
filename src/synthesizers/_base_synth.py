"""
Abstract base class for synthesizers, defining a set of functions each synthesizer must contain
"""

from abc import ABC, abstractmethod

import numpy as np


class Synthesizer(ABC):
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
