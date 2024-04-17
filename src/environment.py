"""
Base environment class, defining a common interface for the agent to interact with synthesizers
Adhering to the OpenAI Gym env naming conventions for functions
"""

import numpy as np
from src.utils.reward_functions import calculate_mse_similarity, time_penalty
from src.utils.audio_processor import AudioProcessor


class Environment:
    def __init__(self, synthesizer=None, control_mode="absolute", render_mode=None, note_length=1.0):
        self.synthesizer = synthesizer
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.note_length = note_length
        self.step_count = 0
        self.target_sound = None
        self.current_sound = None

        # TODO: is it bad practice to automatically call reset() on init?
        # reset environment to prevent NoneType error when reset() is not explicitly called by user before step()
        self.reset()

    def reset(self):
        """
        Resets the environment to a (random) initial state and returns the initial observation.
        This function is typically called at the beginning of each new episode

        :return: state
        """
        self.step_count = 0
        self.target_sound = self.play_sound_random_params()
        self.current_sound = self.play_sound_random_params()

        state = self.calculate_state()

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
        print(f"DEBUG ACTIONS: {action}")
        # Environment is either controlled by incremental changes to synth parameters, or by directly setting them
        if self.control_mode == "incremental":
            current_params = self.get_synth_params()
            new_params = current_params + action
            self.set_synth_params(new_params)

        elif self.control_mode == "absolute":
            self.set_synth_params(action)

        else:
            raise ValueError("control_mode must be either 'incremental' or 'absolute'")

        self.step_count += 1

        self.current_sound = self.synthesizer.play_note(note='C4', duration=self.note_length)

        state = self.calculate_state()
        reward = self.reward_function()
        done = self.check_if_done()

        if done:
            print(f"DEBUG FINAL PARAMS:{self.get_synth_params()}")

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
            raise Exception("Render method called without specifying any render mode.")

        pass

    def get_num_params(self):
        if self.synthesizer is None:
            raise Exception("get_num_params is called without loading a synthesizer.")

        return self.synthesizer.num_params

    def set_synth_params(self, parameters: np.ndarray):
        """
        Iteratively set all synth parameter values
        :param parameters: list of synthesizer parameter values
        :return:
        """
        for index, value in enumerate(parameters):
            self.synthesizer.set_param_value(index, value)
        return

    def get_synth_params(self) -> np.ndarray:
        """
        Iterate over all the synth parameters to get their current value
        :return: list of current synthesizer parameter values
        """
        return np.array([self.synthesizer.get_param_value(i) for i in range(self.synthesizer.num_params)])

    def play_sound_random_params(self):
        # Randomize parameters and play a sound
        param_len = self.get_num_params()
        random_params = np.random.uniform(low=0.0, high=1.0, size=param_len)
        print(f"DEBUG: PARAMETERS: {random_params}")
        self.set_synth_params(parameters=random_params)

        return self.synthesizer.play_note(note='C4', duration=self.note_length)

    def calculate_state(self):
        # FIXME: PLACEHOLDER CODE -- idea: have a state definition, instead of just returning the current sound
        target_audio = AudioProcessor(audio_sample=self.target_sound, sampling_freq=44100.0).calculate_spectrogram()
        current_audio = AudioProcessor(audio_sample=self.current_sound, sampling_freq=44100.0).calculate_spectrogram()
        return np.expand_dims(target_audio - current_audio, axis=-1)

    def reward_function(self):
        # FIXME: PLACEHOLDER CODE -- properly pass audio processor object between functions and classes
        #   maybe instantiate self.target- and self.current_audio and update the sample on note play, instead of new obj
        target_audio = AudioProcessor(audio_sample=self.target_sound, sampling_freq=44100.0).calculate_spectrogram()
        current_audio = AudioProcessor(audio_sample=self.current_sound, sampling_freq=44100.0).calculate_spectrogram()
        similarity_score = calculate_mse_similarity(
            current_sample=current_audio,
            target_sample=target_audio
        )
        penalty = time_penalty(step_count=self.step_count, beta=0.05)

        reward = similarity_score  # - penalty
        print(f"Step: {self.step_count}")
        print(f"Sim score: {similarity_score}, penalty: {penalty}, reward: {reward}")

        return reward

    def check_if_done(self):
        similarity_score = calculate_mse_similarity(current_sample=self.current_sound, target_sample=self.target_sound)
        if similarity_score >= 0.99:  # or similarity_score <= 1e-5:
            return True

        if self.step_count > 100:
            return True

        return False
