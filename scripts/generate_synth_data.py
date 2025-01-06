import os
import tensorflow as tf
from src.environment.environment import Environment
from src.synthesizers import Host, SimpleSynth
from tqdm import tqdm

# Set constants
SAMPLING_RATE = 44100.0
NOTE_LENGTH = 0.5
NUM_SAMPLES = 500000
BATCH_SIZE = 1000


def serialize_example(spectrogram, param_error):
    """
    Convert a single (spectrogram, param_error) pair into a serialized TFRecord Example.
    """
    feature = {
        'spectrogram': tf.train.Feature(float_list=tf.train.FloatList(value=spectrogram.flatten())),
        'param_error': tf.train.Feature(float_list=tf.train.FloatList(value=param_error.flatten())),
        'spectrogram_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=spectrogram.shape)),
        'param_error_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=param_error.shape)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'labeled_spectrograms')
    os.makedirs(data_dir, exist_ok=True)

    # Create synthesizer objects
    host = Host(synthesizer=SimpleSynth, sample_rate=SAMPLING_RATE)
    env = Environment(
        synth_host=host,
        note_length=NOTE_LENGTH,
        control_mode="absolute",
        render_mode=None,
        sampling_freq=SAMPLING_RATE
    )

    # Determine shape of one sample
    env.reset(increment_episode=False)
    dummy_spec = env.calculate_state(form="stacked_spectrogram")
    dummy_err = env.target_params - env.current_params
    spectrogram_shape = dummy_spec.shape
    param_shape = dummy_err.shape

    print(f"Spectrogram shape: {spectrogram_shape}, Parameter error shape: {param_shape}")

    # Write TFRecords
    tfrecord_path = os.path.join(data_dir, 'SimpleSynth_original.tfrecords')
    print(f"Writing TFRecords to {tfrecord_path} ...")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
        for _ in tqdm(range(num_batches), desc="Generating Data"):
            # Initialize batch arrays
            spectrogram_batch = []
            param_error_batch = []

            for _ in range(BATCH_SIZE):
                env.reset(increment_episode=False)
                spectrogram = env.calculate_state(form="stacked_spectrogram")
                param_error = env.calculate_state(form="synth_param_error")
                spectrogram_batch.append(spectrogram)
                param_error_batch.append(param_error)

            # Write the batch
            for spectrogram, param_error in zip(spectrogram_batch, param_error_batch):
                example = serialize_example(spectrogram, param_error)
                writer.write(example)

    print("TFRecords creation complete.")


if __name__ == "__main__":
    main()
