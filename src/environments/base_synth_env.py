"""
Base synthesizer environment class, defining a common interface
Adhering to the OpenAI Gym env
"""

from abc import ABC, abstractmethod


class SynthEnv(ABC):
    def __init__(self):
        pass  # Initialize shared attributes or configurations

    @abstractmethod
    def reset(self):
        """
        Resets the environment to a (random) initial state and returns the initial observation.
        This function is typically called at the beginning of each new episode

        :return: state
        """
        pass

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
        pass
