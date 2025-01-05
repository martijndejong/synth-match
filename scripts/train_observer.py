import os
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

import wandb

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from src.observers.spectrogram_observer import build_spectrogram_observer
from src.utils.config_manager import Config


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"epoch": epoch, "training_loss": logs.get("loss"), "validation_loss": logs.get("val_loss")})


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
    # -------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config()
    config_path = os.path.join(script_dir, "configs", "train_observer.yaml")
    config.load(config_path)

    # Store basic settings from config
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    val_fraction = config["training"]["val_fraction"]

    # -------------------------------------------------------------------
    # 2. Initialize wandb
    # -------------------------------------------------------------------
    wandb.init(
        project=config["experiment"]["project_name"],
        name=config["experiment"]["run_name"],
        group=config["experiment"]["group"],
        config=config
    )

    # -------------------------------------------------------------------
    # 3. File paths
    # -------------------------------------------------------------------
    data_dir = os.path.join(script_dir, config["data"]["directory"])
    tfrecord_path = os.path.join(data_dir, config["data"]["tfrecord_filename"])

    # -------------------------------------------------------------------
    # 4. Count number of records (if not provided in config)
    # -------------------------------------------------------------------
    num_samples = config["training"].get("num_samples", None)

    if num_samples is None:
        # If num_samples is not provided, count records in the TFRecord file
        print("Counting records in the TFRecord file...")
        num_samples = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
        print(f"Total records found: {num_samples}")

    # -------------------------------------------------------------------
    # 5. Create the base TF Dataset
    # -------------------------------------------------------------------
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # -------------------------------------------------------------------
    # 6. Train/Validation split
    # -------------------------------------------------------------------
    val_size = int(num_samples * val_fraction)
    train_size = num_samples - val_size

    # Shuffle once before splitting
    dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=False)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Shuffle again for training each epoch
    train_dataset = train_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    # -------------------------------------------------------------------
    # 7. Batch, Repeat, and Prefetch
    # -------------------------------------------------------------------
    train_dataset = train_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------
    # 8. Build/Compile the Model
    # -------------------------------------------------------------------
    sample_spec, sample_err = next(iter(train_dataset))
    spectrogram_shape = sample_spec.shape[1:]  # excluding batch dimension
    num_params = sample_err.shape[1]

    observer_network = build_spectrogram_observer(
        input_shape=spectrogram_shape,
        feature_dim=config["observer"]["feature_dim"],
        num_params=num_params,
        include_output_layer=True
    )
    observer_network.compile(
        optimizer=Adam(learning_rate=config["observer"]["learning_rate"]),
        loss='mean_squared_error'
    )

    # -------------------------------------------------------------------
    # 9. Prepare model saving
    # -------------------------------------------------------------------
    base_dir = os.path.join(script_dir, config["model_saving"]["base_dir"])
    run_name = config["experiment"]["run_name"]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder = f"{time_stamp}_{run_name}"
    model_save_dir = os.path.join(base_dir, run_folder)
    os.makedirs(model_save_dir, exist_ok=True)

    model_weights_path = os.path.join(model_save_dir, "observer_weights.h5")

    checkpoint_callback = ModelCheckpoint(
        filepath=model_weights_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # -------------------------------------------------------------------
    # 10. Train the Model
    # -------------------------------------------------------------------
    print("Training observer network...")
    history = observer_network.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=train_size // batch_size,
        validation_data=val_dataset,
        validation_steps=val_size // batch_size,
        callbacks=[checkpoint_callback, CustomWandbCallback()],
        verbose=1
    )

    # -------------------------------------------------------------------
    # 11. Save the config that was used
    # -------------------------------------------------------------------
    config_copy_path = os.path.join(model_save_dir, "config_used.yaml")
    config.save(config_copy_path)
    print(f"[INFO] Saved copy of config to {config_copy_path}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
