import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src.observers.spectrogram_observer import build_spectrogram_observer
import matplotlib.pyplot as plt
import os
import h5py


def main():
    # Set constants
    batch_size = 64
    epochs = 10

    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'data/labeled_spectrograms' directory
    data_dir = os.path.join(script_dir, '..', 'data', 'labeled_spectrograms')

    # Path to HDF5 file
    h5_path = os.path.join(data_dir, 'SimpleSynth.h5')

    # Open HDF5 file
    h5f = h5py.File(h5_path, 'r')

    # Get dataset shapes
    num_samples = h5f['stacked_spectrograms'].shape[0]
    spectrogram_shape = h5f['stacked_spectrograms'].shape[1:]
    num_params = h5f['param_errors'].shape[1]

    # Build observer network with output layer
    input_shape = spectrogram_shape  # Shape excluding batch dimension
    observer_network = build_spectrogram_observer(
        input_shape=input_shape,
        num_params=num_params,
        include_output_layer=True
    )

    # Compile model
    observer_network.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error'
    )

    # Create indices for splitting
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    val_fraction = 0.1
    val_size = int(num_samples * val_fraction)
    train_size = num_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Function to retrieve data from HDF5 file
    def get_data(idx):
        idx = idx.numpy()
        spectrogram = h5f['stacked_spectrograms'][idx]
        param_error = h5f['param_errors'][idx]
        return spectrogram, param_error

    # Create datasets using from_tensor_slices
    train_dataset = tf.data.Dataset.from_tensor_slices(train_indices)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_indices)

    # Map the indices to data samples using tf.py_function
    train_dataset = train_dataset.map(
        lambda idx: tf.py_function(
            func=get_data, inp=[idx], Tout=(tf.float32, tf.float32)
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_dataset = val_dataset.map(
        lambda idx: tf.py_function(
            func=get_data, inp=[idx], Tout=(tf.float32, tf.float32)
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Shuffle and batch the training dataset
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Batch and prefetch the validation dataset
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Directory to save the observer network weights
    save_dir = os.path.join(script_dir, '..', 'saved_models', 'observer')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'SimpleSynth.h5')

    # Set up the model checkpoint callback to save the model only when validation loss improves
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Train model
    print("Training observer network...")
    history = observer_network.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback]
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Close the HDF5 file
    h5f.close()


if __name__ == "__main__":
    main()
