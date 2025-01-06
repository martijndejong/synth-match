"""
Audio processing class for feature extraction, normalization, etc.
"""
import librosa
import numpy as np


class AudioProcessor:
    def __init__(self, audio_sample: np.ndarray, sampling_freq: float):
        self.sampling_freq = sampling_freq
        self.audio_sample = audio_sample
        self.spectrogram = None

    def update_sample(self, audio_sample):
        self.audio_sample = audio_sample
        self.convert_to_mono()
        self.crop_sample(length=1.0)
        self.calculate_spectrogram()
        self.normalize()

    def convert_to_mono(self):
        if self.audio_sample.shape[0] == 2:
            self.audio_sample = self.audio_sample.mean(axis=0)
        return

    def crop_sample(self, length: float):
        """
        Crop audio sample to specified length in seconds.
        Add 0 padding if new width longer than original sample.
        """
        new_len = int(length * self.sampling_freq)
        cur_len = len(self.audio_sample)
        if new_len > cur_len:
            self.audio_sample = np.pad(self.audio_sample, (0, new_len - cur_len))
        else:
            self.audio_sample = self.audio_sample[:new_len]
        return

    def normalize(self, db_floor=-80.0, db_ceil=0.0):
        """
        Normalize the self.spectrogram to [0,1] based on a fixed dB range.
        db_floor: The lower bound in dB (e.g., -80 dB).
        db_ceil: The upper bound in dB (e.g., 0 dB).
        """
        # Here self.spectrogram is currently in dB after calculate_spectrogram
        # Range: [db_floor, db_ceil] -> [0,1]
        self.spectrogram = (self.spectrogram - db_floor) / (db_ceil - db_floor)
        self.spectrogram = np.clip(self.spectrogram, 0, 1)

    def calculate_spectrogram(self, hop_len: int = 512, n_mels: int = 256):
        # Compute mel-scaled spectrogram
        S_mel = librosa.feature.melspectrogram(
            y=self.audio_sample,
            sr=self.sampling_freq,
            hop_length=hop_len,
            n_mels=n_mels
        )

        # Instead of using ref=np.max, use a fixed reference amplitude (ref=1.0)
        S_db_mel = librosa.amplitude_to_db(S_mel, ref=1.0)

        # Now self.spectrogram is in absolute dB terms relative to amplitude=1.
        self.spectrogram = S_db_mel
        return
