# Import environment and synthesizer
from src.environment import Environment
from src.synthesizers.super_simple_synth import SuperSimpleSynth

# Import observer and actor network
from src.observers.waveform_observer import build_waveform_observer
from src.agents.actor_critic_agent import ActorCriticAgent

from src.utils.replay_buffer import ReplayBuffer
import numpy as np

# Set constants
SAMPLING_RATE = 44100.0
NOTE_LENGTH = 0.5

# Create synthesizer object
synth = SuperSimpleSynth(sample_rate=SAMPLING_RATE)

# Create environment object and pass synthesizer object
env = Environment(synthesizer=synth, note_length=NOTE_LENGTH, control_mode="incremental")

action_dim = env.get_num_params()
hidden_dim = 256  # Can be adjusted
gamma = 0.99  # Discount factor for future rewards

# Create Observer network and Actor Critic agent network
observer_network = build_waveform_observer(
    input_shape=(int(SAMPLING_RATE*NOTE_LENGTH), 1)
)
model = ActorCriticAgent(
    observer_network=observer_network,
    action_dim=action_dim,
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
