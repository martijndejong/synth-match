"""
At this moment, the train_agent.py script is basically a complete copy of the train_end_to_end script.
The only differences are that, in this script:
 - the observer network is frozen. Such that the agent can form a policy while receiving consistent features
 - only the agent weights are saved after training, instead of the full model object
"""
import os
import wandb
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Environment & Synth
from src.environment.environment import Environment
from src.synthesizers import Host, SimpleSynth
from pyvst import SimpleHost

# Observer & Agent
from src.observers import build_spectrogram_observer
from src.agents import TD3Agent

from src.utils.config_manager import Config
from src.utils.replay_buffer import ReplayBuffer


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1) Load config for end-to-end training ---
    config = Config()
    config_path = os.path.join(script_dir, "configs", "train_agent.yaml")
    config.load(config_path)

    # --- 2) Initialize wandb (separate from observer script) ---
    wandb.init(
        project=config["experiment"]["project_name"],
        name=config["experiment"]["run_name"],
        group=config["experiment"]["group"],
        config=config
    )

    # --- 3) Create environment & synthesizer ---
    # If you have a specific VST plugin path:
    # host = SimpleHost("/path/to/vst.so", sample_rate=sampling_rate)
    # Otherwise, using our custom synthesizer and wrapper:
    host = Host(synthesizer=SimpleSynth, sample_rate=config["environment"]["sampling_rate"])

    env = Environment(
        synth_host=host,
        note_length=config["environment"]["note_length"],
        control_mode=config["environment"]["control_mode"],
        render_mode=config["environment"]["render_mode"],
        sampling_freq=config["environment"]["sampling_rate"]
    )

    input_shape = env.get_input_shape()
    output_shape = env.get_output_shape()

    # --- 4) Build or load the observer network ---
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

    # --- 5) Create the agent (TD3) ---
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

    # --- 7) Initialize replay buffer ---
    replay_memory = ReplayBuffer(capacity=config["replay_buffer"]["capacity"])

    batch_size = config["replay_buffer"]["batch_size"]
    num_episodes = config["training"]["num_episodes"]

    # --- 8) Training loop ---
    rewards_mem = []

    for episode in tqdm(range(num_episodes), desc="Training agent on simulated experiences"):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Store experience
            replay_memory.add(
                (state, action, reward, next_state, done)
            )

            # Train if buffer is large enough
            if len(replay_memory) > batch_size:
                batch = replay_memory.sample(batch_size)
                states, actions, rewards, next_states, dones = map(
                    np.array, zip(*batch)
                )
                agent.train_step((states, actions, rewards, next_states, dones))

            state = next_state

        print(f"[INFO] Episode {episode + 1}, Reward: {episode_reward:.2f}")
        wandb.log({"episode": episode + 1, "reward": episode_reward})
        rewards_mem.append(episode_reward)

    # --- 9) Systematic saving of final weights ---
    base_dir = os.path.join(script_dir, config["model_saving"]["base_dir"])
    # Use run_name + timestamp for a subfolder
    run_name = config["experiment"]["run_name"]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder = f"{time_stamp}_{run_name}"

    model_save_dir = os.path.join(base_dir, run_folder)
    os.makedirs(model_save_dir, exist_ok=True)

    agent.save_actor_critic_weights(model_save_dir)
    print(f"[INFO] Saved agent weights to {model_save_dir}")

    # Save config used
    config_copy_path = os.path.join(model_save_dir, "config_used.yaml")
    config.save(config_copy_path)
    print(f"[INFO] Saved copy of config to {config_copy_path}")

    # We don't store big artifacts on wandb (to save memory),
    print("[INFO] End-to-end training completed.")
    wandb.finish()


if __name__ == "__main__":
    main()
