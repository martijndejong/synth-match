# Import environment and synthesizer
from src.environment.environment import Environment
from src.synthesizers import Host, SimpleSynth
from pyvst import SimpleHost

# Import observer and actor network
from src.observers import build_spectrogram_observer
from src.agents import TD3Agent

from src.utils.replay_buffer import ReplayBuffer
import numpy as np

import os
import matplotlib.pyplot as plt

from tqdm import tqdm
import time

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the '../saved_models' directory
saved_models_dir = os.path.join(script_dir, '..', 'saved_models')


# Set constants
SAMPLING_RATE = 44100.0
NOTE_LENGTH = 0.5

# Create synthesizer object
host = Host(synthesizer=SimpleSynth, sample_rate=SAMPLING_RATE)
# host = SimpleHost("/mnt/c/github/synth-match/amsynth_vst.so", sample_rate=SAMPLING_RATE)

# Create environment object and pass synthesizer object
env = Environment(synth_host=host, note_length=NOTE_LENGTH, control_mode="incremental", render_mode=False, sampling_freq=SAMPLING_RATE)

hidden_dim = 256
gamma = 0.9
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2

input_shape = env.get_input_shape()
output_shape = env.get_output_shape()

# Create the observer network
observer_network = build_spectrogram_observer(
    input_shape=input_shape,
    include_output_layer=False  # Exclude the output layer used during observer pre-training
)
# Load weights from the pre-trained observer network
observer_weights_path = f'{saved_models_dir}/observer/SimpleSynth.h5'
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
agent.observer_network.trainable = True

# # Load pre-trained actor and critic weights
actor_weights_path = f'{saved_models_dir}/end_to_end/actor_weights.h5'
critic_weights_path = f'{saved_models_dir}/end_to_end/critic_weights.h5'
agent.load_actor_critic_weights(actor_weights_path, critic_weights_path)

# Initialize replay memory
replay_memory = ReplayBuffer(capacity=int(1e5))
batch_size = 128  # Batch size for training from replay memory

num_episodes = 2000  # Number of episodes to train
start_time = time.time()  # Initialize timer

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

    # Log time every 10 episodes
    if episode % 10 == 0:
        time_elapsed = time.time() - start_time
        print(f"Episode {episode}: Time elapsed = {time_elapsed:.2f} seconds")

print(rewards_mem)

# Closing and saving results
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
save_dir = os.path.join(script_dir, '..', 'saved_models', 'end_to_end')
os.makedirs(save_dir, exist_ok=True)

# # Save model weights
# actor_save_path = os.path.join(save_dir, 'actor_weights.h5')
# critic_save_path = os.path.join(save_dir, 'critic_weights.h5')
# agent.save_actor_critic_weights(actor_save_path, critic_save_path)
# print(f"Actor weights saved to {actor_save_path}")
# print(f"Critic weights saved to {critic_save_path}")

# Plotting
# Plot total rewards per episode, and a rolling average (smoother)
plt.figure()

# Plot raw rewards
plt.plot(rewards_mem, label='Reward per Episode')

# Compute rolling average over last 50 episodes
window_size = 50
rolling_averages = []
for i in range(len(rewards_mem)):
    start_idx = max(0, i - window_size + 1)
    window = rewards_mem[start_idx:i+1]
    rolling_averages.append(np.mean(window))

# Plot rolling average
plt.plot(rolling_averages, label=f'Rolling Avg (last {window_size})')

plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.savefig(os.path.join(save_dir, 'rewards_per_episode.png'))
plt.show()
