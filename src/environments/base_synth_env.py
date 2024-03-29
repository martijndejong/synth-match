"""
Base synthesizer environment class, defining a common interface
Adhering to the OpenAI Gym env naming conventions for functions
"""

from abc import ABC, abstractmethod


class SynthEnv(ABC):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        pass  # Initialize shared attributes or configurations

    @abstractmethod
    def reset(self):
        """
        Resets the environment to a (random) initial state and returns the initial observation.
        This function is typically called at the beginning of each new episode

        :return: state
        """
        state = None

        if self.render_mode == "human":
            self.render()
        return state

    @abstractmethod
    def step(self, action):
        """
        The core function where the environment reacts to an action taken by the agent.
        It updates the environment's state, calculates the reward, and determines whether the episode has ended

        :param action: output of agent, i.e., synthesizer parameters passed to play note
        :return: state, reward, done
        state: The new state of the environment after applying the action
        reward: A numerical value indicating the reward obtained from taking the action
        done: boolean indicating whether the episode has ended
        """
        state = None
        reward = 0.0
        done = False

        if self.render_mode == "human":
            self.render()
        return state, reward, done

    def render(self):
        """
        Let's add an optional rendering function that can be turned on/off. This would use, e.g., Matplotlib to
        visualize (animate) the incoming/outgoing spectrogram/audio waves/setting values/etc.
        This could be used for debugging or for display of trained network capabilities.
        """
        if self.render_mode is None:
            raise Exception("Render method called without specifying any render mode")

        pass
