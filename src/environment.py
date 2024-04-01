"""
Base environment class, defining a common interface for the agent to interact with synthesizers
Adhering to the OpenAI Gym env naming conventions for functions
"""

import numpy as np


class Environment:
    def __init__(self, synthesizer=None, control_mode="absolute", render_mode=None):
        self.synthesizer = synthesizer
        self.control_mode = control_mode
        self.render_mode = render_mode
        pass  # Initialize shared attributes or configurations

    def reset(self):
        """
        Resets the environment to a (random) initial state and returns the initial observation.
        This function is typically called at the beginning of each new episode

        :return: state
        """
        param_len = len(self.synthesizer.get_parameters())
        # TODO: EACH PARAMETER SHOULD HAVE ITS OWN BOUNDS, THESE SHOULD BE USED TO GENERATE SENSIBLE RANDOM PARAM VALUES
        #   THESE BOUNDS WILL ALSO BE IMPORTANT FOR THE AGENT TO KNOW
        random_params = np.random.uniform(low=0.0, high=1.0, size=param_len)

        self.synthesizer.set_parameters(parameters=random_params)

        state = self.synthesizer.play_note(note='C4', duration=2.0)  # FIXME: SHOULD DURATION BE AN ENV INIT PARAM?

        if self.render_mode == "human":
            self.render()
        return state

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

        # TODO: SHOULD THE CONTROL MODE 'ABSOLUTE' VS 'INCREMENTAL' BE PART OF THE AGENT, OR OF THE ENVIRONMENT?
        if self.control_mode == "incremental":
            current_params = self.synthesizer.get_parameters()
            new_params = current_params + action
            self.synthesizer.set_parameters(new_params)

        elif self.control_mode == "absolute":
            self.synthesizer.set_parameters(action)

        else:
            raise ValueError("control_mode must be either 'incremental' or 'absolute'")

        state = self.synthesizer.play_note(note='C8', duration=2.0)  # FIXME: SHOULD DURATION BE AN ENV INIT PARAM?
        reward = 0
        done = False

        if self.render_mode == "human":
            self.render()
        return state, reward, done

    def render(self):
        """
        Let's add an optional rendering function that can be turned on/off. This would use, e.g., Matplotlib to
        visualize (animate) the incoming/outgoing spectrogram/audio waves/setting values/etc.
        This could be used for debugging or for display of trained network capabilities.
        The render function could also turn on/off audio playback
        """
        if self.render_mode is None:
            raise Exception("Render method called without specifying any render mode")

        pass
