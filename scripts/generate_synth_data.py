import numpy as np
import os
import h5py
from src.environment.environment import Environment
from src.synthesizers import Host, SimpleSynth
from tqdm import tqdm

# Set constants
SAMPLING_RATE = 44100.0
NOTE_LENGTH = 0.5
NUM_SAMPLES = 500000  # Total number of samples generated
BATCH_SIZE = 1000  # Number of samples per batch


def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create synthesizer object
    host = Host(synthesizer=SimpleSynth, sample_rate=SAMPLING_RATE)

    # Create environment object and pass synthesizer object
    env = Environment(
        synth_host=host,
        note_length=NOTE_LENGTH,
        control_mode="absolute",
        render_mode=None,
        sampling_freq=SAMPLING_RATE
    )

    # Determine the shape of one sample to initialize datasets
    env.reset(increment_episode=False)
    # Generate a sample state
    stacked_spectrogram = env.calculate_state(form="stacked_spectrogram")
    param_error = env.target_params - env.current_params

    # Get shapes
    spectrogram_shape = stacked_spectrogram.shape  # (height, width, channels)
    param_shape = param_error.shape  # (num_params,)

    # Construct the path to the 'data/labeled_spectrograms' directory
    data_dir = os.path.join(script_dir, '..', 'data', 'labeled_spectrograms')
    os.makedirs(data_dir, exist_ok=True)

    # Create HDF5 file
    h5_path = os.path.join(data_dir, 'SimpleSynth.h5')
    with h5py.File(h5_path, 'w') as h5f:
        # Create datasets with appropriate shapes
        spectrograms_ds = h5f.create_dataset(
            'stacked_spectrograms',
            shape=(NUM_SAMPLES,) + spectrogram_shape,
            maxshape=(NUM_SAMPLES,) + spectrogram_shape,
            chunks=(BATCH_SIZE,) + spectrogram_shape,
            dtype=np.float32,
            compression=None  # Disable compression for faster writes
        )
        param_errors_ds = h5f.create_dataset(
            'param_errors',
            shape=(NUM_SAMPLES,) + param_shape,
            maxshape=(NUM_SAMPLES,) + param_shape,
            chunks=(BATCH_SIZE,) + param_shape,
            dtype=np.float32,
            compression=None  # Disable compression for faster writes
        )

        # Generate data and write to datasets in batches
        print("Generating data and writing to HDF5 file...")
        num_batches = NUM_SAMPLES // BATCH_SIZE
        if NUM_SAMPLES % BATCH_SIZE != 0:
            num_batches += 1  # Include the last partial batch

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, NUM_SAMPLES)
            current_batch_size = end_idx - start_idx

            print(f"Processing batch {batch_idx + 1}/{num_batches} "
                  f"({start_idx}-{end_idx})")

            # Initialize arrays to hold the batch data
            spectrograms_batch = np.zeros(
                (current_batch_size,) + spectrogram_shape, dtype=np.float32)
            param_errors_batch = np.zeros(
                (current_batch_size,) + param_shape, dtype=np.float32)

            # Generate data for the batch
            for i in range(current_batch_size):
                # Reset environment to generate new target and current sounds
                env.reset(increment_episode=False)
                # Get the stacked spectrogram (input)
                stacked_spectrogram = env.calculate_state(form="stacked_spectrogram")
                # Get the parameter error (output)
                param_error = env.calculate_state(form="synth_param_error")

                spectrograms_batch[i] = stacked_spectrogram
                param_errors_batch[i] = param_error

            # Write the batch to the datasets
            spectrograms_ds[start_idx:end_idx] = spectrograms_batch
            param_errors_ds[start_idx:end_idx] = param_errors_batch

    print(f"Data saved to {os.path.abspath(h5_path)}")


if __name__ == "__main__":
    main()
