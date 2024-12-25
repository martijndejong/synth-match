"""
(!) TURNS OUT THIS IS LESS EFFICIENT THAN TRAINING ON PRE-GENERATED DATA WITH EPOCHS
    because generating the spectrogram samples is the bottleneck

train_observer_live.py

Continuously generates training data for the observer network in chunks and trains on it.
Stops either when no sufficient improvement is seen or when MAX_SAMPLES is processed in one session.
Allows pausing/resuming by saving checkpoints (network weights + metadata).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import environment and synthesizer
from src.environment.environment import Environment
from src.synthesizers import Host, SimpleSynth
from src.observers.spectrogram_observer import build_spectrogram_observer

# -------------------------------------------------------------------------
# User-Defined Hyperparameters
# -------------------------------------------------------------------------
NOTE_LENGTH = 0.5  # Duration of each note in seconds
SAMPLING_RATE = 44100.0  # Audio sampling rate
CHUNK_SIZE = 10000  # Number of new samples generated per iteration
BATCH_SIZE = 64  # Batch size for training
MAX_SAMPLES = 500000  # Max total samples for this script run
IMPROVEMENT_FACTOR = 0.999  # Stop if current_loss >= best_loss * IMPROVEMENT_FACTOR


# Save folder for observer network weights
def get_save_dir(script_dir):
    """
    Returns the path to the 'saved_models/observer' directory, consistent with existing code.
    """
    return os.path.join(script_dir, '..', 'saved_models', 'observer')


CHECKPOINT_PREFIX = "observer_live"
META_FILENAME = f"{CHECKPOINT_PREFIX}_meta.npz"
WEIGHTS_FILENAME = f"{CHECKPOINT_PREFIX}_weights.h5"


def main():
    """
    Main function to train the observer network by continuously generating new samples.
    Stops when IMPROVEMENT_FACTOR is not met or MAX_SAMPLES is processed in this run.
    """
    # -------------------------------------------------------------------------
    # 1. Initialization & Checkpoints
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = get_save_dir(script_dir)
    os.makedirs(save_dir, exist_ok=True)

    meta_path = os.path.join(save_dir, META_FILENAME)
    ckpt_path = os.path.join(save_dir, WEIGHTS_FILENAME)

    # Create environment with hardcoded control mode
    host = Host(synthesizer=SimpleSynth, sample_rate=SAMPLING_RATE)
    env = Environment(
        synth_host=host,
        note_length=NOTE_LENGTH,
        control_mode="absolute",
        render_mode=None,
        sampling_freq=SAMPLING_RATE
    )

    # Retrieve input/output shapes from environment
    input_shape = env.get_input_shape()  # e.g., stacked spectrogram shape
    output_shape = env.get_output_shape()  # e.g., number of synthesizer parameters

    print("[INFO] Environment shapes:")
    print(f"       input_shape={input_shape}, output_shape={output_shape}")

    # Build observer network with final layer (predict param_error)
    observer_network = build_spectrogram_observer(
        input_shape=input_shape,
        num_params=output_shape,
        include_output_layer=True
    )
    observer_network.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error'
    )

    # Initialize metadata
    chunk_index = 0
    best_loss = np.inf
    losses_history = []

    # For plotting a line if we resume
    old_session_len = 0

    # Attempt to load existing checkpoint
    if os.path.exists(meta_path) and os.path.exists(ckpt_path):
        print("[INFO] Found existing checkpoint. Loading...")
        meta_data = np.load(meta_path, allow_pickle=True)
        chunk_index = int(meta_data["chunk_index"])
        best_loss = float(meta_data["best_loss"])
        losses_history = list(meta_data["losses_history"])
        old_session_len = len(losses_history)

        observer_network.load_weights(ckpt_path)
        print(f"[INFO] Resuming from chunk_index={chunk_index}, best_loss={best_loss:.6f}")
    else:
        print("[INFO] No checkpoint found. Starting fresh training...")

    # Compute how many chunks we will train in this run
    # Each chunk processes CHUNK_SIZE new samples, up to MAX_SAMPLES in total
    num_chunks = int(np.ceil(MAX_SAMPLES / CHUNK_SIZE))

    # -------------------------------------------------------------------------
    # 2. Main Training Loop (for each chunk)
    # -------------------------------------------------------------------------
    print("[INFO] Starting live training loop...")
    total_generated = 0
    stop_flag = False

    for i in tqdm(range(chunk_index + 1, num_chunks + 1), desc="Live Observer Training"):
        # a) Determine how many samples left to generate
        samples_left = MAX_SAMPLES - total_generated
        if samples_left <= 0:
            print("[INFO] Already generated the maximum samples for this run.")
            break
        current_chunk_size = min(CHUNK_SIZE, samples_left)

        # b) Generate chunk_size samples
        spectrograms_batch, param_errors_batch = generate_chunk(env, current_chunk_size)
        total_generated += current_chunk_size

        # c) Train on this chunk
        history = observer_network.fit(
            x=spectrograms_batch,
            y=param_errors_batch,
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            shuffle=False  # Already random data
        )
        final_loss = history.history["loss"][-1]
        losses_history.append(final_loss)
        print(f"[INFO] Chunk {i}: chunk_size={current_chunk_size}, training_loss={final_loss:.6f}")

        # d) Check improvement
        if final_loss < best_loss * IMPROVEMENT_FACTOR:
            best_loss = final_loss
            # Save checkpoint
            observer_network.save_weights(ckpt_path)
            np.savez(
                meta_path,
                chunk_index=i,
                best_loss=best_loss,
                losses_history=losses_history
            )
            print(f"[INFO] Improved! best_loss={best_loss:.6f}. Checkpoint saved (chunk={i}).")
        else:
            # Not enough improvement
            print(
                f"[INFO] No significant improvement. current_loss={final_loss:.6f}, threshold={best_loss * IMPROVEMENT_FACTOR:.6f}")
            stop_flag = True

        # Clean up memory
        del spectrograms_batch
        del param_errors_batch

        if stop_flag:
            print("[INFO] Early stopping triggered.")
            break

    # -------------------------------------------------------------------------
    # 3. Final Check & Plot
    # -------------------------------------------------------------------------
    if len(losses_history) > 0:
        final_loss = losses_history[-1]
        print(f"[INFO] Final training loss: {final_loss:.6f}")
    else:
        final_loss = np.inf
        print("[WARN] No training was performed? losses_history is empty.")

    if final_loss < best_loss:
        observer_network.save_weights(ckpt_path)
        np.savez(
            meta_path,
            chunk_index=num_chunks,
            best_loss=final_loss,
            losses_history=losses_history
        )
        print("[INFO] Final checkpoint saved with updated best loss.")

    # Plot the entire loss history
    plot_loss_history(losses_history, old_session_len)

    print(f"[INFO] Training completed. Processed ~{total_generated} samples this run. Best Loss={best_loss:.6f}")


def generate_chunk(env, chunk_size):
    """
    Generate a chunk of training data by resetting the environment multiple times.
    Returns:
      spectrograms_batch (np.ndarray): shape=(chunk_size, *env.get_input_shape())
      param_errors_batch (np.ndarray): shape=(chunk_size, env.get_output_shape())
    """
    input_shape = env.get_input_shape()
    output_shape = env.get_output_shape()
    spectrograms_batch = np.zeros((chunk_size,) + input_shape, dtype=np.float32)
    param_errors_batch = np.zeros((chunk_size, output_shape), dtype=np.float32)

    for i in tqdm(range(chunk_size), desc="Generating spectrograms", position=0, leave=True):
        env.reset(increment_episode=False)
        spectrograms_batch[i] = env.calculate_state(form="stacked_spectrogram")
        # "synth_param_error" => target_params - current_params
        param_errors_batch[i] = env.calculate_state(form="synth_param_error")

    return spectrograms_batch, param_errors_batch


def plot_loss_history(loss_history, old_session_len):
    """
    Plots the full training loss history.
    Adds a vertical red line if resuming from a previous session.
    """
    if len(loss_history) == 0:
        print("[WARN] No losses to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss')
    if 0 < old_session_len < len(loss_history):
        # Draw a vertical line to indicate where the new session started
        plt.axvline(x=old_session_len, color='r', linestyle='--', label='Resumed Here')
    plt.title('Observer Live Training Loss')
    plt.xlabel('Chunk Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
