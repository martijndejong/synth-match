"""
Base environment class, defining a common interface for the agent to interact with synthesizers
Adhering to the OpenAI Gym env naming conventions for functions
"""

import numpy as np
from src.environment.reward_functions import rmse_similarity, time_cost, action_cost
from src.utils.audio_processor import AudioProcessor
from src.environment.render_functions import initialize_plots, update_plots


class Environment:
    def __init__(self, synth_host=None, control_mode="absolute", render_mode=None, note_length=1.0, sampling_freq=44100.0):
        self.synth_host = synth_host
        self.synthesizer = self.synth_host.vst
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.note_length = note_length
        self.sampling_freq = sampling_freq

        self.current_audio = AudioProcessor(audio_sample=None, sampling_freq=self.sampling_freq)
        self.target_audio = AudioProcessor(audio_sample=None, sampling_freq=self.sampling_freq)

        self.episode = 0
        self.step_count = 0
        self.last_reward = 0
        self.total_reward = 0

        self.last_action = None
        self.state = None
        self.target_sound = None
        self.target_params = None
        self.current_sound = None
        self.current_params = None
        self.previous_params = None
        self.param_names = self.get_synth_param_names()

        # Create empty figure for render function
        if render_mode:
            self.fig, self.axes = initialize_plots(rows=2, cols=2)

    def get_output_shape(self):
        """
        Provide the output shape for the Agent (i.e., number of actions)
        """
        return self.get_num_params()
    
    def get_input_shape(self):
        """
        Provide the input shape for the Agent (i.e., state shape)
        """
        rnd_state = self.reset(increment_episode=False)
        return rnd_state.shape
    
    def reset(self, increment_episode = True):
        """
        Resets the environment to a (random) initial state and returns the initial observation.
        This function is typically called at the beginning of each new episode

        :return: state
        """
        if increment_episode:
            self.episode += 1
        self.step_count = 0
        self.total_reward = 0
        self.target_sound, self.target_params = self.play_sound_random_params()
        self.current_sound, self.current_params = self.play_sound_random_params()

        self.target_audio.update_sample(audio_sample=self.target_sound)
        self.current_audio.update_sample(audio_sample=self.current_sound)

        state = self.calculate_state()

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
        self.last_action = action
        # Environment is either controlled by incremental changes to synth parameters, or by directly setting them
        self.previous_params = self.get_synth_params()  # store synth params before taking step
        if self.control_mode == "incremental":
            new_params = self.previous_params + action
            self.set_synth_params(new_params)

        elif self.control_mode == "absolute":
            self.set_synth_params(action)

        else:
            raise ValueError("control_mode must be either 'incremental' or 'absolute'")
        self.current_params = self.get_synth_params()  # store synth params after taking step

        self.step_count += 1

        self.current_sound = self.synth_host.play_note(note=64, note_duration=self.note_length)
        self.current_audio.update_sample(audio_sample=self.current_sound)

        # Update state, reward, and done, after step
        state = self.calculate_state()
        reward = self.reward_function()
        done = self.check_if_done()

        # Set environment attributes
        self.state = state
        self.last_reward = reward
        self.total_reward += reward

        if done:
            print(f"DEBUG FINAL PARAMS:{self.get_synth_params()}")

        if self.render_mode:
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

        update_plots(
            env=self
        )

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

    def get_synth_param_names(self) -> list[str]:
        """
        Iterate over all the synth parameters to get their names
        :return: list of synthesizer parameter names
        """
        return [self.synthesizer.get_param_name(i) for i in range(self.synthesizer.num_params)]

    def play_sound_random_params(self):
        # Randomize parameters and play a sound
        param_len = self.get_num_params()
        random_params = np.random.uniform(low=0.0, high=1.0, size=param_len)
        self.set_synth_params(parameters=random_params)

        return self.synth_host.play_note(note=64, note_duration=self.note_length), random_params

    def calculate_state(self):
        # return np.expand_dims(self.target_audio.spectrogram - self.current_audio.spectrogram, axis=-1)
        # return np.stack((self.current_audio.spectrogram, self.target_audio.spectrogram), axis=-1)
        return self.target_params - self.current_params

    def reward_function(self):
        # FIXME: PLACEHOLDER CODE -- properly pass audio processor object between functions and classes
        target_audio = self.target_audio.spectrogram
        current_audio = self.current_audio.spectrogram

        similarity_score = rmse_similarity(current_sample=self.current_params, target_sample=self.target_params)
        # similarity_score = rmse_similarity(current_sample=current_audio, target_sample=target_audio)
        time_penalty = time_cost(step_count=self.step_count, factor=0.01)
        action_penalty = action_cost(
            action=self.last_action,
            factor=10.0
        )

        reward = similarity_score - time_penalty - action_penalty
        print(f"DEBUG REWARD: {reward:.3f} = {similarity_score:.3f} - {time_penalty:.3f} - {action_penalty:.3f}")

        return reward

    def check_if_done(self):
        # TODO: There should be a big reward if the episode is ended by high similarity score
        similarity_score = rmse_similarity(current_sample=self.current_audio.spectrogram, target_sample=self.target_audio.spectrogram)
        if similarity_score >= 0.99:  # or similarity_score <= 1e-5:
            return True

        if self.step_count > 100:
            return True

        return False
