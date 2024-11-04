import numpy as np
import os
import matplotlib.pyplot as plt

# Import environment and synthesizer
from src.environment.environment import Environment
from src.synthesizers.super_simple_synth import SuperSimpleHost

# Import the dummy observer network
from src.observers.parameter_observer import build_parameter_observer

# Import the agent network
from src.agents import TD3Agent

# Import the replay buffer
from src.utils.replay_buffer import ReplayBuffer

from tqdm import tqdm


def main():
    # Set constants
    SAMPLING_RATE = 44100.0
    NOTE_LENGTH = 1.0

    # Create synthesizer object
    host = SuperSimpleHost(sample_rate=SAMPLING_RATE)

    # Create environment object and pass synthesizer object
    env = Environment(
        synth_host=host,
        note_length=NOTE_LENGTH,
        control_mode="incremental",
        render_mode=False,
        sampling_freq=SAMPLING_RATE,
        default_state_form="synth_param_error"  # Return parameter error as state
    )

    hidden_dim = 64
    gamma = 0.9
    tau = 0.005
    policy_noise = 0.3
    noise_clip = 0.5
    policy_delay = 3

    # Get input and output shapes
    input_shape = env.get_input_shape()  # Should be (num_params,)
    output_shape = env.get_output_shape()  # Number of synthesizer parameters

    # Create the dummy observer network
    observer_network = build_parameter_observer(input_shape=input_shape)

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

    # Initialize replay memory
    replay_memory = ReplayBuffer(capacity=10000)
    batch_size = 64  # Batch size for training from replay memory

    num_episodes = 500  # Number of episodes to train

    rewards_mem = []      # For logging total rewards per episode
    actor_losses = []     # For logging actor losses
    critic_losses = []    # For logging critic losses

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        synth_params = env.get_synth_params()
        done = False
        episode_reward = 0

        while not done:
            # Agent acts based on the current state (parameter error) and current synth parameters
            action = agent.act(state, synth_params)

            # Environment step
            next_state, reward, done = env.step(action)
            next_synth_params = env.get_synth_params()
            episode_reward += reward

            # Store experience in replay memory
            replay_memory.add((state, synth_params, action, reward, next_state, next_synth_params, done))

            # If enough samples are available in memory, sample a batch and perform a training step
            if len(replay_memory) > batch_size:
                # print("Training step")
                sampled_experiences = replay_memory.sample(batch_size)
                states, synth_params_batch, actions, rewards, next_states, next_synth_params_batch, dones = map(
                    np.array, zip(*sampled_experiences))
                loss = agent.train_step(
                    (states, synth_params_batch, actions, rewards, next_states, next_synth_params_batch, dones))
                # Collect losses
                actor_losses.append(loss['actor_loss'])
                critic_losses.append(loss['critic_loss'])

            state = next_state
            synth_params = next_synth_params

        print(f'Episode {episode + 1}, Total Reward: {episode_reward:.2f}')
        rewards_mem.append(episode_reward)
    print(rewards_mem)

    # After training is complete
    # Save the trained agent's actor and critic weights
    save_dir = os.path.join('..', 'saved_models', 'agent')
    os.makedirs(save_dir, exist_ok=True)
    actor_save_path = os.path.join(save_dir, 'actor_weights.h5')
    critic_save_path = os.path.join(save_dir, 'critic_weights.h5')
    agent.save_actor_critic_weights(actor_save_path, critic_save_path)
    print(f"Actor weights saved to {actor_save_path}")
    print(f"Critic weights saved to {critic_save_path}")

    # Plotting
    # Plot total rewards per episode
    plt.figure()
    plt.plot(rewards_mem)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(save_dir, 'rewards_per_episode.png'))
    plt.show()

    # Plot actor and critic losses
    plt.figure()
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.title('Losses over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.show()


if __name__ == "__main__":
    main()
