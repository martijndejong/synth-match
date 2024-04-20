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

    def convert_to_mono(self):
        pass

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

        self.spectrogram = S_db_mel

        # FIXME: RETURN SPECTOGRAM? OR KEEP IT AS CLASS ATTRIBUTE THAT IS FILLED?
        #   PLUS MAKE NORMALIZATION MORE CONTROLLED INSTEAD OF HARDCODED IN RETURN HERE
        return (S_db_mel + 80)/80
