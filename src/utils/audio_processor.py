"""
Audio processing class for feature extraction, normalization, etc.
"""
from scipy import signal
from dataclasses import dataclass
import numpy as np


class AudioProcessor:
    def __init__(self, audio_sample: np.ndarray, sampling_freq: int):
        self.sampling_freq = sampling_freq
        self.audio_sample = audio_sample
        self.spectrogram = None

    def calculate_spectrogram(self):
        # Process self.audio_sample to extract features
        # Store in self.features or return the features
        f, t, Sxx = signal.spectrogram(self.audio_sample, self.sampling_freq)
        self.spectrogram = Spectrogram(frequency=f, time=t, magnitude=Sxx)

    def normalize(self):
        # Normalize self.audio_sample
        pass


@dataclass
class Spectrogram:
    frequency: np.ndarray
    time: np.ndarray
    magnitude: np.ndarray
