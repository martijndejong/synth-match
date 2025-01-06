import os
import h5py
import tensorflow as tf
from tqdm import tqdm

# Set constants
BATCH_SIZE = 10000  # Adjust based on available memory and HDF5 chunk size


def _serialize_example(spectrogram, param_error):
    """
    Serialize a single (spectrogram, param_error) pair into a TFRecord Example.
    """
    feature = {
        'spectrogram': tf.train.Feature(float_list=tf.train.FloatList(value=spectrogram.flatten())),
        'param_error': tf.train.Feature(float_list=tf.train.FloatList(value=param_error.flatten())),
        'spectrogram_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=spectrogram.shape)),
        'param_error_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=param_error.shape)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def hdf5_to_tfrecords(h5_path, tfrecord_path, batch_size):
    """
    Convert an HDF5 file to a TFRecord file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file (input).
    tfrecord_path : str
        Path to save the TFRecord file (output).
    batch_size : int
        Number of samples to process at a time (for memory efficiency).
    """
    # Open the HDF5 file
    with h5py.File(h5_path, 'r') as h5f:
        # Get datasets
        spectrograms = h5f['stacked_spectrograms']
        param_errors = h5f['param_errors']

        num_samples = spectrograms.shape[0]
        print(f"Converting {num_samples} samples from HDF5 to TFRecords...")

        # Write to TFRecords
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for start_idx in tqdm(range(0, num_samples, batch_size)):
                end_idx = min(start_idx + batch_size, num_samples)

                # Load a batch from HDF5
                spectrogram_batch = spectrograms[start_idx:end_idx]
                param_error_batch = param_errors[start_idx:end_idx]

                # Serialize and write each sample
                for spectrogram, param_error in zip(spectrogram_batch, param_error_batch):
                    example = _serialize_example(spectrogram, param_error)
                    writer.write(example)

    print(f"TFRecords saved to {tfrecord_path}")


def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'data/labeled_spectrograms' directory
    data_dir = os.path.join(script_dir, '..', 'data', 'labeled_spectrograms')

    # Path to HDF5 file
    h5_path = os.path.join(data_dir, 'SimpleSynth.h5')

    # Path to output TFRecord file
    tfrecord_path = os.path.join(data_dir, 'SimpleSynth.tfrecords')

    # Convert HDF5 to TFRecords
    hdf5_to_tfrecords(h5_path, tfrecord_path, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
