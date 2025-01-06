from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model


def build_spectrogram_observer(input_shape=(128, 256, 2), feature_dim=128, num_params=None, include_output_layer=False):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(feature_dim, activation='relu')(x)  # This is the feature layer - ALIGN SIZE WITH AGENT INPUT
    x = Dropout(0.5)(x)

    if include_output_layer and num_params is not None:
        # Output layer for predicting parameter error, used for pre-training
        outputs = Dense(num_params, activation='tanh')(x)
    else:
        outputs = x

    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    observer_network = build_spectrogram_observer()
    observer_network.summary()
