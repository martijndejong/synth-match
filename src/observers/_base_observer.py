import tensorflow as tf


class BaseObserver(tf.keras.Model):
    """Base class for all observer networks."""
    def call(self, inputs):
        raise NotImplementedError("Each observer must implement the call method.")