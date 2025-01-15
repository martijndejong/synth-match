from src.environment.environment import Environment
from src.synthesizers import Host, SimpleSynth
from src.observers import build_spectrogram_observer
from src.agents import TD3Agent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.audio_processor import AudioProcessor
import os
import numpy as np
import json
import yaml

def initialize_environment(config):
    # Create synthesizer and environment
    host = Host(synthesizer=SimpleSynth, sample_rate=config["environment"]["sampling_rate"])
    env = Environment(
        synth_host=host,
        note_length=config["environment"]["note_length"],
        control_mode=config["environment"]["control_mode"],
        render_mode=config["environment"]["render_mode"],
        sampling_freq=config["environment"]["sampling_rate"]
    )
    return env

def initialize_agent(env, config):
    # Load script_dir | TODO - not repeat this code
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_shape = env.get_input_shape()
    output_shape = env.get_output_shape()

    observer_network = build_spectrogram_observer(
        input_shape=input_shape,
        include_output_layer=False  # Exclude output layer for end-to-end
    )
    # If we load weights from a previously trained observer
    if config["model_loading"]["load_observer_weights"]:
        observer_path = os.path.join(script_dir, "..", "saved_models", "observer")
        observer_subfolder = config["model_loading"]["observer_subfolder"]

        observer_weights_path = os.path.join(observer_path, observer_subfolder, "observer_weights.h5")
        observer_network.load_weights(observer_weights_path, by_name=True, skip_mismatch=True)
        print(f"[INFO] Loaded observer weights from {observer_weights_path}")
    
    # Create the agent (TD3)
    agent = TD3Agent(
        observer_network=observer_network,
        action_dim=output_shape,
        hidden_dim=config["agent"]["hidden_dim"],
        gamma=config["agent"]["gamma"],
        tau=config["agent"]["tau"],
        policy_noise=config["agent"]["policy_noise"],
        noise_clip=config["agent"]["noise_clip"],
        policy_delay=config["agent"]["policy_delay"]
    )
    # If you want the observer to be trainable or frozen, set appropriately:
    agent.observer_network.trainable = config["training"]["trainable_observer"]
    
    # --- 6) Optionally load pre-trained actor/critic weights ---
    if config["model_loading"]["load_pretrained_agent"]:
        agent_path = os.path.join(script_dir, "..", "saved_models", "agent")
        agent_subfolder = config["model_loading"]["agent_subfolder"]

        load_dir = os.path.join(agent_path, agent_subfolder)
        agent.load_actor_critic_weights(load_dir)
        print(f"[INFO] Loaded pretrained agent from {load_dir}")
    
    return agent

def match_random_sound(env, agent):
    """
    Matches the target sound using the given environment and agent.
    """
    state = env.reset()
    synth_params = env.get_synth_params()
    done = False

    while not done:
        action = agent.act(state, synth_params)
        next_state, reward, done = env.step(action)
        synth_params = env.get_synth_params()
        state = next_state

    return env

def convert_numpy_to_audio_data(audio_wave, sample_rate, bit_depth):
    # Normalize the audio wave to [-1.0, 1.0] range (if not already)
    max_val = np.max(np.abs(audio_wave))
    normalized_wave = audio_wave / max_val if max_val > 0 else audio_wave
    
    # Scale to integer range for the given bit depth
    max_int = 2**(bit_depth - 1) - 1
    scaled_wave = (normalized_wave * max_int).astype(np.int16)  # 16-bit signed integers
    
    # Convert to list for JSON serialization
    samples = scaled_wave.tolist()
    
    # Create the target audio data structure
    target_audio_data = {
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "samples": samples
    }
    
    return target_audio_data


def convert_to_spectrogram_json(spectrogram_matrix):
    # Convert to list for JSON serialization
    pixel_values = spectrogram_matrix.tolist()

    # Extract dimensions
    height, width = spectrogram_matrix.shape

    # Create the target spectrogram data structure
    target_spectrogram_data = {
        "width": width,
        "height": height,
        "pixel_values": pixel_values
    }

    return target_spectrogram_data


def format_target_json_from_state(env):
    '''
    Formats the target audio and spectrogram data into JSON format to be used by the API call
    '''

    # Extract fields from environment
    target_audio = env.target_audio.audio_sample
    target_spectrogram = env.target_audio.spectrogram
    sampling_freq = env.sampling_freq

    # Concert to required fields | TODO - change hardoded bit_depth
    target_audio_data = convert_numpy_to_audio_data(target_audio, sample_rate=sampling_freq, bit_depth=16)
    target_spectrogram_data = convert_to_spectrogram_json(target_spectrogram)

    # Combine both into the required format
    target_data = {
        "target_audio_data": target_audio_data,
        "target_spectrogram_data": target_spectrogram_data
    }

    return target_data

def format_matched_parameters(host_parameter_values, host_parameter_names, mapping_file):
    """
    Combines parameter names and values into a dictionary using a mapping loaded from a YAML file.

    Args:
        values (list): A list of parameter values.
        parameters (list): A list of parameter names.
        mapping_file (str): Path to the YAML file containing the parameter mapping.

    Returns:
        str: A JSON-formatted string with the combined parameters.
    """
    # Convert values to Python native types (e.g., float32 to float)
    host_parameter_values = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in host_parameter_values]

    # Load the mapping from the YAML file
    with open(mapping_file, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Extract the mapping dictionary
    mapping = yaml_data["synth_parameter_mapping"]

    # Combine the values and mapped parameter names into a dictionary
    matched_parameters = {
        mapping[param]: value for param, value in zip(host_parameter_names, host_parameter_values)
    }

    return matched_parameters


def format_match_json_from_state(env, mapping_file):
    '''
    Formats the match/current audio, spectrogram & parameter data into JSON format to be
    used by the API call
    '''

    # Extract fields from environment
    current_audio = env.current_audio.audio_sample
    current_spectrogram = env.current_audio.spectrogram
    sampling_freq = env.sampling_freq
    host_parameter_values = env.current_params
    host_parameter_names = env.param_names

    # Convert subfields to API format | TODO - change hardoded bit_depth
    matched_parameters = format_matched_parameters(host_parameter_values, host_parameter_names, mapping_file)
    matched_audio_data = convert_numpy_to_audio_data(current_audio, sample_rate=sampling_freq, bit_depth=16)
    matched_spectrogram_data = convert_to_spectrogram_json(current_spectrogram)

    # Combine both into the required format
    match_data = {
        "matched_parameters": matched_parameters,
        "matched_audio_data": matched_audio_data,
        "matched_spectrogram_data": matched_spectrogram_data
    }

    return match_data