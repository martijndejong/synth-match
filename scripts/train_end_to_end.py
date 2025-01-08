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
from src.utils.math import vector_lerp


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1) Load config for end-to-end training ---
    config = Config()
    config_path = os.path.join(script_dir, "configs", "train_end_to_end.yaml")
    config.load(config_path)

    # --- 2) Initialize wandb ---
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

    # --- 8) Training loop ---
    rewards_mem = []

    for episode in tqdm(range(config["training"]["num_episodes"]), desc="Training agent on simulated experiences"):
        state = env.reset()
        synth_params = env.get_synth_params()
        done = False
        episode_reward = 0.0

        # Compute expert_fraction (expert assistance) for this episode
        if config["expert_correction"]["enabled"]:
            fraction = 1.0 - (episode / config["expert_correction"]["decay_episodes"])
            fraction = max(0.0, min(1.0, fraction))  # clamp between [0, 1]
            expert_fraction = config["expert_correction"]["start_expert_fraction"] * fraction
        else:
            expert_fraction = 0.0

        while not done:
            agent_action = agent.act(state, synth_params)

            # If expert correction is enabled, compute perfect action and blend into agent action
            if expert_fraction > 0.0:
                expert_action = env.calculate_state(form="synth_param_error")
                action = vector_lerp(agent_action, expert_action, expert_fraction)
            else:
                action = agent_action

            # Step in the environment
            next_state, reward, done = env.step(action)
            next_synth_params = env.get_synth_params()
            episode_reward += reward

            # Store experience
            replay_memory.add(
                (state, synth_params, action, reward, next_state, next_synth_params, done)
            )

            # Train if buffer is large enough
            if len(replay_memory) > batch_size:
                batch = replay_memory.sample(batch_size)
                states, synths, actions, rewards, nxt_states, nxt_synths, dones = map(
                    np.array, zip(*batch)
                )
                agent.train_step((states, synths, actions, rewards,
                                  nxt_states, nxt_synths, dones))

            state = next_state
            synth_params = next_synth_params

        print(f"[INFO] Episode {episode + 1}, Reward: {episode_reward:.2f}")

        # Log to wandb
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "expert_fraction": expert_fraction
        })

        rewards_mem.append(episode_reward)

    # --- 9) Save final weights, config, etc. ---
    base_dir = os.path.join(script_dir, "..", "saved_models", "end_to_end")
    run_name = config["experiment"]["run_name"]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder = f"{time_stamp}_{run_name}"

    model_save_dir = os.path.join(base_dir, run_folder)
    os.makedirs(model_save_dir, exist_ok=True)

    # FIXME: Saving model does not work, potential fix: Unify the call signature (multiple inputs)
    # def call(self, inputs, training=False):
    #     """
    #     inputs is a tuple: (states, synth_params)
    #     or a dict: {"states": ..., "synth_params": ...}
    #     """
    #     states, synth_params = inputs

    # agent.save_end_to_end(model_save_dir)
    # print(f"[INFO] Saved model object to {model_save_dir}")

    # Save config used
    config_copy_path = os.path.join(model_save_dir, "config_used.yaml")
    config.save(config_copy_path)
    print(f"[INFO] Saved copy of config to {config_copy_path}")

    print("[INFO] End-to-end training completed.")
    wandb.finish()


if __name__ == "__main__":
    main()
