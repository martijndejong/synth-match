import os
import tensorflow as tf
from datetime import datetime

import wandb

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# Reuse the same parse function you used in train_observer.py
# (Or copy the parse_tfrecord function if it's slightly different.)
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


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": logs.get("loss"),
            "validation_loss": logs.get("val_loss")
        })


def build_actor_pretrain_model(observer_network, actor, input_shape):
    """
    Build a small Keras model:
        Input: spectrogram
        Output: actor(observer(spectrogram))

    We freeze the observer's weights, but keep the actor trainable.
    """
    # Freeze observer
    observer_network.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="spectrogram_input")
    features = observer_network(inputs)  # shape = (batch, feature_dim)
    outputs = actor(features)  # shape = (batch, action_dim)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pretrain_actor_model")
    return model


def main():
    from src.utils.config_manager import Config
    from src.observers.spectrogram_observer import build_spectrogram_observer
    from src.agents.td3_agent import TD3Agent  # We'll reuse its actor

    # -------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config()
    config_path = os.path.join(script_dir, "configs", "pretrain_actor.yaml")
    config.load(config_path)

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
    # 7. Batch, Repeat, Prefetch
    # -------------------------------------------------------------------
    train_dataset = train_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------
    # 8. Build the Observer + Actor
    # -------------------------------------------------------------------
    # We'll build the same observer as before (including_output_layer=False if
    # you want only features). Then load the weights if needed.
    sample_spec, sample_err = next(iter(train_dataset))
    spectrogram_shape = sample_spec.shape[1:]  # excluding batch
    action_dim = sample_err.shape[1]  # param_error dimension

    # 1) Observer
    observer_network = build_spectrogram_observer(
        input_shape=spectrogram_shape,
        feature_dim=config["observer"]["feature_dim"],
        num_params=None,  # No final dense layer for param_error, just features
        include_output_layer=False
    )
    # Optionally load observer weights from disk if you have them
    observer_weights_path = config["observer"].get("pretrained_weights_path", None)
    if observer_weights_path and os.path.exists(observer_weights_path):
        observer_network.load_weights(
            observer_weights_path,
            by_name=True,
            skip_mismatch=True
        )
        print(f"[INFO] Loaded observer weights from {observer_weights_path}")

    # 2) Actor (from TD3Agent)
    # We instantiate a TD3Agent purely to get its actor; we won't use critics here.
    temp_agent = TD3Agent(
        observer_network=observer_network,
        action_dim=action_dim
    )
    # We want to freeze observer and only train the actor part, so let's build
    # a sub-model that outputs actor(features).
    pretrain_model = build_actor_pretrain_model(
        observer_network=temp_agent.observer_network,
        actor=temp_agent.actor,
        input_shape=spectrogram_shape
    )

    pretrain_model.compile(
        optimizer=Adam(learning_rate=1e-4),  # or from config if you prefer
        loss="mean_squared_error"
    )

    # -------------------------------------------------------------------
    # 9. Prepare model saving (actor-only)
    # -------------------------------------------------------------------
    base_dir = os.path.join(script_dir, config["model_saving"]["base_dir"])
    run_name = config["experiment"]["run_name"]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder = f"{time_stamp}_{run_name}"
    model_save_dir = os.path.join(base_dir, run_folder)
    os.makedirs(model_save_dir, exist_ok=True)

    # We'll save the actor weights here.
    actor_weights_path = os.path.join(model_save_dir, "actor_weights.h5")

    # Standard ModelCheckpoint would save the entire sub-model's weights (including observer).
    # We only want to save the actor weights, so let's do a custom callback:
    class SaveActorWeights(tf.keras.callbacks.Callback):
        def __init__(self, actor, filepath, monitor='val_loss', mode='min', save_best_only=True):
            super().__init__()
            self.actor = actor
            self.filepath = filepath
            self.monitor = monitor
            self.mode = mode
            self.save_best_only = save_best_only
            self.best_val = None
            if mode == 'min':
                self.best_val = float('inf')
            elif mode == 'max':
                self.best_val = -float('inf')

        def on_epoch_end(self, epoch, logs=None):
            current_val = logs.get(self.monitor)
            if current_val is None:
                return

            improved = False
            if self.mode == 'min' and current_val < self.best_val:
                improved = True
                self.best_val = current_val
            elif self.mode == 'max' and current_val > self.best_val:
                improved = True
                self.best_val = current_val

            if not self.save_best_only or improved:
                # Save only the actor weights
                self.actor.save_weights(self.filepath, overwrite=True)
                print(f"[INFO] Actor weights saved to {self.filepath}")

    save_actor_callback = SaveActorWeights(
        actor=temp_agent.actor,
        filepath=actor_weights_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # -------------------------------------------------------------------
    # 10. Train the Model
    # -------------------------------------------------------------------
    steps_per_epoch = train_size // batch_size
    val_steps = val_size // batch_size

    print("Pretraining actor network (supervised) ...")
    history = pretrain_model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=[save_actor_callback, CustomWandbCallback()],
        verbose=1
    )

    # -------------------------------------------------------------------
    # 11. Save the final config used
    # -------------------------------------------------------------------
    config_copy_path = os.path.join(model_save_dir, "config_used.yaml")
    config.save(config_copy_path)
    print(f"[INFO] Saved copy of config to {config_copy_path}")

    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()
