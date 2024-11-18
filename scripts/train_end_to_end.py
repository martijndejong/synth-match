# Import environment and synthesizer
from src.environment.environment import Environment
from src.synthesizers.super_simple_synth import SuperSimpleHost
from pyvst import SimpleHost

# Import observer and actor network
from src.observers import build_spectrogram_observer
from src.agents import TD3Agent

from src.utils.replay_buffer import ReplayBuffer
import numpy as np

import os
import matplotlib.pyplot as plt

from tqdm import tqdm

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the '../saved_models' directory
saved_models_dir = os.path.join(script_dir, '..', 'saved_models')


# Set constants
SAMPLING_RATE = 44100.0
NOTE_LENGTH = 1.0

# Create synthesizer object
host = SuperSimpleHost(sample_rate=SAMPLING_RATE)
# host = SimpleHost("/mnt/c/github/synth-match/amsynth_vst.so", sample_rate=SAMPLING_RATE)

# Create environment object and pass synthesizer object
env = Environment(synth_host=host, note_length=NOTE_LENGTH, control_mode="incremental", render_mode=False, sampling_freq=SAMPLING_RATE)

hidden_dim = 128
gamma = 0.9
tau = 0.005
policy_noise = 0.3
noise_clip = 0.5
policy_delay = 4

input_shape = env.get_input_shape()
output_shape = env.get_output_shape()

# Create the observer network
observer_network = build_spectrogram_observer(
    input_shape=input_shape,
    include_output_layer=False  # Exclude the output layer used during observer pre-training
)
# Load weights from the pre-trained observer network
observer_weights_path = f'{saved_models_dir}/observer/SuperSimpleSynth.h5'
observer_network.load_weights(observer_weights_path, by_name=True, skip_mismatch=True)

# Create the Actor-Critic agent
agent = TD3Agent(
    observer_network=observer_network,
    action_dim=output_shape,
    hidden_dim=hidden_dim,
    gamma=gamma,
    tau=tau,
    policy_noise=policy_noise,
    noise_clip=noise_clip,
    policy_delay=policy_delay
)

# Freeze the observer network
agent.observer_network.trainable = False

# Load pre-trained actor and critic weights
actor_weights_path = f'{saved_models_dir}/agent/actor_weights.h5'
critic_weights_path = f'{saved_models_dir}/agent/critic_weights.h5'
agent.load_actor_critic_weights(actor_weights_path, critic_weights_path)

# Initialize replay memory
replay_memory = ReplayBuffer(capacity=10000)
batch_size = 64  # Batch size for training from replay memory

num_episodes = 1000  # Number of episodes to train

rewards_mem = []  # TODO: replace by more systematic logging system in utility functions
for episode in tqdm(range(num_episodes)):
    state = env.reset()
    synth_params = env.get_synth_params()
    done = False
    episode_reward = 0

    while not done:
        action = agent.act(state, synth_params)
        next_state, reward, done = env.step(action)
        next_synth_params = env.get_synth_params()
        episode_reward += reward

        # Store experience in replay memory
        replay_memory.add((state, synth_params, action, reward, next_state, next_synth_params, done))

        # If enough samples are available in memory, sample a batch and perform a training step
        if len(replay_memory) > batch_size:
            # print("Training step")
            sampled_experiences = replay_memory.sample(batch_size)
            states, synth_params, actions, rewards, next_states, next_synth_paramss, dones = map(np.array, zip(*sampled_experiences))
            agent.train_step((states, synth_params, actions, rewards, next_states, next_synth_paramss, dones))

        state = next_state
        synth_params = next_synth_params

    print(f'Episode {episode + 1}, Total Reward: {episode_reward:.2f}')
    rewards_mem.append(episode_reward)
print(rewards_mem)

# Plotting
# Plot total rewards per episode
plt.figure()
plt.plot(rewards_mem)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
save_dir = os.path.join('..', 'saved_models', 'end_to_end')
plt.savefig(os.path.join(save_dir, 'rewards_per_episode.png'))
plt.show()
