import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from src.observers.spectrogram_observer import build_spectrogram_observer


def parse_tfrecord(serialized_example):
    """
    Parse a single TFRecord into (spectrogram, param_error) Tensors.
    """
    feature_spec = {
        'spectrogram': tf.io.VarLenFeature(tf.float32),
        'param_error': tf.io.VarLenFeature(tf.float32),
        'spectrogram_shape': tf.io.VarLenFeature(tf.int64),
        'param_error_shape': tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_spec)

    # Convert sparse to dense
    spectrogram_flat = tf.sparse.to_dense(parsed['spectrogram'])
    param_error_flat = tf.sparse.to_dense(parsed['param_error'])
    spectrogram_shape = tf.sparse.to_dense(parsed['spectrogram_shape'])
    param_error_shape = tf.sparse.to_dense(parsed['param_error_shape'])

    # Reshape tensors to their original shapes
    spectrogram = tf.reshape(spectrogram_flat, spectrogram_shape)
    param_error = tf.reshape(param_error_flat, param_error_shape)

    return spectrogram, param_error


def main():
    # ---------------------------------------
    # 1. Basic settings
    # ---------------------------------------
    batch_size = 64
    epochs = 10
    val_fraction = 0.1

    # ---------------------------------------
    # 2. File paths
    # ---------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'labeled_spectrograms')
    tfrecord_path = os.path.join(data_dir, 'SimpleSynth.tfrecords')

    # ---------------------------------------
    # 3. Inspect the total number of samples
    #    (Assumes you know how many records were written.)
    # ---------------------------------------
    # If you don't know exactly how many records are in the file, you can count them:
    # print("Going to count records")
    # num_samples = 0
    # for _ in tf.data.TFRecordDataset(tfrecord_path):
    #     num_samples += 1
    # print("Total records found:", num_samples)
    num_samples = 500000
    steps_per_epoch = num_samples // batch_size

    # ---------------------------------------
    # 4. Create the base TF Dataset
    # ---------------------------------------
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # We'll parse each record into (spectrogram, param_error).
    # Choose the parse function that matches how you wrote your TFRecords:
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    # or dataset = dataset.map(_parse_tfrecord_legacy, num_parallel_calls=tf.data.AUTOTUNE)

    # ---------------------------------------
    # 5. Train/Validation split using skip/take
    # ---------------------------------------
    val_size = int(num_samples * val_fraction)
    train_size = num_samples - val_size

    # Because it's a single file, and we haven't shuffled yet, let's shuffle first
    # (so that val_dataset is actually random).
    # Shuffle the entire dataset - we can set buffer_size to num_samples or 10000 etc.
    # Then "re-batch" into train/val by skipping/taking.
    dataset = dataset.shuffle(buffer_size=min(num_samples, 10000), reshuffle_each_iteration=False)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Now we can shuffle again for training each epoch:
    train_dataset = train_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    # ---------------------------------------
    # 6. Batch and Prefetch
    # ---------------------------------------
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # ---------------------------------------
    # 7. Build/Compile the Model
    #    (We can infer shape from a sample.)
    # ---------------------------------------
    # Peek at 1 element to get shape
    sample_spec, sample_err = next(iter(train_dataset))
    spectrogram_shape = sample_spec.shape[1:]  # excluding batch dimension
    num_params = sample_err.shape[1]

    observer_network = build_spectrogram_observer(
        input_shape=spectrogram_shape,
        num_params=num_params,
        include_output_layer=True
    )
    observer_network.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error'
    )

    # ---------------------------------------
    # 8. Model Checkpoint callback
    # ---------------------------------------
    save_dir = os.path.join(script_dir, '..', 'saved_models', 'observer')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'SimpleSynth_TEMPDELETE.h5')

    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # ---------------------------------------
    # 9. Train the Model
    # ---------------------------------------
    print("Training observer network...")
    history = observer_network.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_size // batch_size,
        callbacks=[checkpoint_callback],
        verbose=1
    )

    # ---------------------------------------
    # 10. Plot Loss Curves
    # ---------------------------------------
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


if __name__ == "__main__":
    main()
