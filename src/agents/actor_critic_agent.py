"""
TODO: below code is currently rough placeholder code and needs to be properly implemented, suggested actions:
    * init of ActorCritic (child class) must create observer ("Parameter 'observer' unfilled")
    * observer network must somehow know the correct input shape and output size (see warning in train_model.py)
    * 'critic' currently assumed as one step, might need more complex interaction (e.g., state-value pair passed)
"""

import tensorflow as tf
from src.agents.base_agent import BaseAgent


class ActorCriticAgent(BaseAgent):
    def __init__(self, continuous_action_size, discrete_action_sizes):
        super(ActorCriticAgent, self).__init__()

        # Define actor layers
        self.actor_continuous = tf.keras.layers.Dense(continuous_action_size, activation='tanh')  # Adjust activation based on parameter ranges
        self.actor_discretes = [tf.keras.layers.Dense(size, activation='softmax') for size in discrete_action_sizes]

        # Define critic layers
        self.critic = [...]

    def call(self, inputs):
        state_features = self.observer(inputs)

        # Continuous actions
        continuous_actions = self.actor_continuous(state_features)
        # Discrete actions
        discrete_actions = [layer(state_features) for layer in self.actor_discretes]
        # Critic value
        value = self.critic(state_features)

        return continuous_actions, discrete_actions, value
