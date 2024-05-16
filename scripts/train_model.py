# Import environment and synthesizer
from src.environment.environment import Environment
from src.synthesizers.super_simple_synth import SuperSimpleHost
from pyvst import SimpleHost

# Import observer and actor network
from src.observers.parameter_observer import build_parameter_observer
from src.agents.actor_critic_agent import ActorCriticAgent

from src.utils.replay_buffer import ReplayBuffer
import numpy as np

# TODO: this import should be removed when no longer needed for temp code
from src.utils.audio_processor import AudioProcessor

# Set constants
SAMPLING_RATE = 44100.0
NOTE_LENGTH = 0.5

# Create synthesizer object
host = SuperSimpleHost(sample_rate=SAMPLING_RATE)
# host = SimpleHost("/mnt/c/github/synth-match/amsynth_vst.so", sample_rate=SAMPLING_RATE)

# Create environment object and pass synthesizer object
env = Environment(synth_host=host, note_length=NOTE_LENGTH, control_mode="incremental", render_mode=True, sampling_freq=SAMPLING_RATE)

hidden_dim = 256  # Can be adjusted
gamma = 0.99  # Discount factor for future rewards

input_shape = env.get_input_shape()
output_shape = env.get_output_shape()

# Create Observer network and Actor Critic agent network
observer_network = build_parameter_observer(
    input_shape=input_shape  # (int(SAMPLING_RATE*NOTE_LENGTH), 1)
)
model = ActorCriticAgent(
    observer_network=observer_network,
    action_dim=output_shape,
    hidden_dim=hidden_dim,
    gamma=gamma
)

# Initialize replay memory
replay_memory = ReplayBuffer(capacity=10000)
batch_size = 64  # Batch size for training from replay memory

num_episodes = 500  # Number of episodes to train

rewards_mem = []  # TODO: replace by more systematic logging system in utility functions
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = model.act(state)
        next_state, reward, done = env.step(action)
        episode_reward += reward

        # Store experience in replay memory
        replay_memory.add((state, action, reward, next_state, done))

        # If enough samples are available in memory, sample a batch and perform a training step
        if len(replay_memory) > batch_size:
            print("Training step")
            sampled_experiences = replay_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*sampled_experiences))
            model.train_step((states, actions, rewards, next_states, dones))

        state = next_state

    print(f'Episode {episode + 1}, Total Reward: {episode_reward:.2f}')
    rewards_mem.append(episode_reward)
print(rewards_mem)
