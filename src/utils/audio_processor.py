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
        self.crop_sample(length=2.0)
        self.calculate_spectrogram()

    def convert_to_mono(self):
        if self.audio_sample.shape[0] == 2:
            self.audio_sample = self.audio_sample.mean(axis=0)
            return
        return
    
    def crop_sample(self, length: float):
        """
        Crop audio sample to specified length in seconds. Add 0 padding if new width longer than original sample.
        """
        new_len = int(length * self.sampling_freq)
        cur_len = len(self.audio_sample)
        if new_len > cur_len:
            self.audio_sample = np.pad(self.audio_sample, (0, new_len - cur_len))

        elif new_len <= cur_len:
            self.audio_sample = self.audio_sample[0:new_len]
        return

    def normalize(self):
        # Normalize self.audio_sample
        pass

    def calculate_spectrogram(self, hop_len: int = 512, n_mels: int = 128):
        # Compute mel-scaled spectrogram
        S_mel = librosa.feature.melspectrogram(
            y=self.audio_sample,
            sr=self.sampling_freq,
            hop_length=hop_len,
            n_mels=n_mels
        )
        S_db_mel = librosa.amplitude_to_db(S_mel, ref=np.max)

        # FIXME: MAKE NORMALIZATION MORE CONTROLLED INSTEAD OF HARDCODED IN RETURN HERE
        self.spectrogram = (S_db_mel + 80)/80

        return
