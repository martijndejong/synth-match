"""
An abstract base class which outlines the common functionalities and interfaces for all RL agents,
such as methods for selecting actions, updating policies, and learning from experiences.
"""
import tensorflow as tf


class BaseAgent(tf.keras.Model):
    """Base class for all agents."""
    def __init__(self, observer):
        super(BaseAgent, self).__init__()
        self.observer = observer

    def call(self, inputs):
        # Model's forward pass
        raise NotImplementedError("Each observer must implement the call method.")
