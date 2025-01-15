"""
Base environment class, defining a common interface for the agent to interact with synthesizers
Adhering to the OpenAI Gym env naming conventions for functions
"""

import numpy as np
from dataclasses import dataclass

from src.environment.reward_functions import (
    time_cost,
    action_cost,
    weighted_ssim,
    saturation_penalty,
    euclidean_distance,
    calculate_ssim,
    masked_ssim,
    mse_similarity,
    rmse_similarity,
    mse_similarity_masked,
    rmse_similarity_masked,
    directional_reward_penalty,
)
from src.utils.audio_processor import AudioProcessor
from src.environment.render_functions import initialize_plots, update_plots


class Environment:
    def __init__(self, synth_host=None, control_mode="absolute", render_mode=None, note_length=1.0,
                 sampling_freq=44100.0, default_state_form="stacked_spectrogram"):
        self.synth_host = synth_host
        self.synthesizer = self.synth_host.vst
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.note_length = note_length
        self.sampling_freq = sampling_freq
        self.default_state_form = default_state_form

        self.current_audio = AudioProcessor(audio_sample=None, sampling_freq=self.sampling_freq)
        self.target_audio = AudioProcessor(audio_sample=None, sampling_freq=self.sampling_freq)

        self.episode = 0
        self.step_count = 0
        self.last_reward = 0
        self.total_reward = 0

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

    def reset(self, increment_episode=True, start_params=np.array([])):
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

        # Either start episode with random parameters, or set parameters
        if start_params.size == 0:
            self.current_sound, self.current_params = self.play_sound_random_params()
        else:
            self.current_sound, self.current_params = self.play_sound_set_params(start_params)

        # Update the sample in our audio preprocessor
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
        # print(f"DEBUG ACTIONS: {action}")
        # Environment is either controlled by incremental changes to synth parameters, or by directly setting them
        self.previous_params = self.get_synth_params()  # store synth params before taking step
        if self.control_mode == "incremental":
            new_params = self.previous_params + action
            self.set_synth_params(new_params)

        elif self.control_mode == "absolute":
            self.set_synth_params(action)

        elif self.control_mode == "human":
            for i, param_name in enumerate(self.get_synth_param_names()):
                action[i] = input(f"Set value for {param_name}:")
            self.set_synth_params(action)

        else:
            raise ValueError("control_mode must be either 'incremental', 'absolute', or 'human'")
        self.current_params = self.get_synth_params()  # store synth params after taking step

        self.step_count += 1

        self.current_sound = self.synth_host.play_note(note=64, note_duration=self.note_length)
        self.current_audio.update_sample(audio_sample=self.current_sound)

        # Update state, reward, and done, after step
        state = self.calculate_state()
        reward, done = self.reward_function(action)

        # Set environment attributes
        self.state = state
        self.last_reward = reward
        self.total_reward += reward

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
        return np.array(
            [self.synthesizer.get_param_value(i) for i in range(self.synthesizer.num_params)],
            dtype=np.float32
        )

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

    def play_sound_set_params(self, params):
        # Set parameters and play a sound
        self.set_synth_params(parameters=params)

        return self.synth_host.play_note(note=64, note_duration=self.note_length), params
    
    def set_and_play_target_parameters(self, params):
        '''
        Takes updated target parameters and returns audio & spectrogram
        '''
        # Save current settings to prevent issues with incremental control setting
        current_settings = self.get_synth_params()
        
        # Create target sound
        # self.target_sound, _ = self.play_sound_set_params(params)
        self.set_synth_params(parameters=params)
        self.target_sound = self.synth_host.play_note(note=64, note_duration=self.note_length)
        self.target_params = params

        # Update target audio
        self.target_audio.update_sample(audio_sample=self.target_sound)
        target_spectrogram = self.target_audio.spectrogram

        # Reset synth to 'current' state, to prevent issues with incremental control setting
        self.set_synth_params(parameters=current_settings)

        return self.target_sound, target_spectrogram


    def calculate_state(self, form=None):
        form = form if form else self.default_state_form

        # State: Stack the current spectrogram and the target spectrogram
        if form == "stacked_spectrogram":
            return np.stack((self.current_audio.spectrogram, self.target_audio.spectrogram), axis=-1)

        # State: Target spectrogram minus current spectrogram
        elif form == "spectrogram_error":
            return np.expand_dims(self.target_audio.spectrogram - self.current_audio.spectrogram, axis=-1)

        # (Cheat) State: No need for observation network, just return synth parameter error directly
        elif form == "synth_param_error":
            return self.target_params - self.current_params

        else:
            raise ValueError("form must be either 'stacked_spectrogram', 'spectrogram_error', or 'synth_param_error'")

    def reward_function(self, action):
        # TODO: WE CAN IMPROVE THE REWARD FUNCTION STILL - BELOW ARE SOME EXAMPLES OF REWARD FUNCTIONS WE'RE NOT USING
        # TODO: WEIGHTED_SSIM SEEMED MOST PROMISING, BUT HAS BEEN RETURNING NAN ON LATEST TESTS
        # - masked_ssim
        # - rmse_similarity
        # - rmse_similarity_masked
        # - directional_reward_penalty

        similarity_score = calculate_ssim(self.current_audio.spectrogram, self.target_audio.spectrogram)
        # time_penalty = time_cost(step_count=self.step_count, factor=0.01)
        # action_penalty = action_cost(
        #     action=action,
        #     factor=10.0
        # )
        saturate_penalty = saturation_penalty(synth_params=self.get_synth_params(), actions=action, factor=1.0)
        # similarity_score = euclidean_distance(self.current_params, self.target_params)  # aka parameter_distance

        is_done, bonus = self.check_if_done(similarity_score)  # usually pass similarity_score

        # reward = similarity_score ** 2 * 10 + parameter_distance - time_penalty - action_penalty - saturate_penalty + bonus
        # reward = 2 * parameter_distance + 2 * similarity_score - time_penalty - action_penalty - saturate_penalty + bonus
        # reward = parameter_distance - saturate_penalty + bonus
        reward = similarity_score - saturate_penalty + bonus

        return reward, is_done

    def check_if_done(self, similarity_score):
        max_steps = 100
        if similarity_score >= 0.95:
        # print(similarity_score)
        # similarity_score (= euclidean_stance (inherently squared)) >= -0.01 worked well with SuperSimpleSynth
        # if similarity_score >= -0.1:
            return True, 100 * (max_steps - self.step_count) / max_steps

        if self.step_count >= max_steps:
            return True, 0  # -100

        return False, 0


@dataclass
class State:
    spectrogram: np.ndarray
    synth_params: np.ndarray
