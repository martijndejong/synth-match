# Audio processing class for feature extraction, normalization, etc.

class AudioProcessor:
    def __init__(self, audio_sample):
        self.audio_sample = audio_sample
        self.features = None

    def extract_features(self):
        # Process self.audio_sample to extract features
        # Store in self.features or return the features
        pass

    def normalize(self):
        # Normalize self.audio_sample
        pass