import numpy as np
import os

# Import environment and synthesizer
from src.environment.environment import Environment
from src.synthesizers.super_simple_synth import SuperSimpleHost

# Import the dummy observer network
from src.observers.parameter_observer import build_parameter_observer

# Import the agent network
from src.agents.actor_critic_agent import ActorCriticAgent

# Import the replay buffer
from src.utils.replay_buffer import ReplayBuffer


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
        render_mode=True,
        sampling_freq=SAMPLING_RATE,
        default_state_form="synth_param_error"  # Return parameter error as state
    )

    hidden_dim = 256  # Can be adjusted
    gamma = 0.99  # Discount factor for future rewards

    # Get input and output shapes
    input_shape = env.get_input_shape()  # Should be (num_params,)
    output_shape = env.get_output_shape()  # Number of synthesizer parameters

    num_params = output_shape  # Number of synthesizer parameters

    # Create the dummy observer network
    observer_network = build_parameter_observer(input_shape=input_shape)

    # Create the Actor-Critic agent
    agent = ActorCriticAgent(
        observer_network=observer_network,
        action_dim=output_shape,
        hidden_dim=hidden_dim,
        gamma=gamma
    )

    # Initialize replay memory
    replay_memory = ReplayBuffer(capacity=10000)
    batch_size = 64  # Batch size for training from replay memory

    num_episodes = 500  # Number of episodes to train

    rewards_mem = []  # For logging rewards

    for episode in range(num_episodes):
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
                print("Training step")
                sampled_experiences = replay_memory.sample(batch_size)
                states, synth_params_batch, actions, rewards, next_states, next_synth_params_batch, dones = map(
                    np.array, zip(*sampled_experiences))
                agent.train_step(
                    (states, synth_params_batch, actions, rewards, next_states, next_synth_params_batch, dones))

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


if __name__ == "__main__":
    main()
