import tensorflow as tf
import wandb


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": logs.get("loss"),
            "validation_loss": logs.get("val_loss")
        })


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
