from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model


def build_waveform_observer(input_shape=(88200, 1)):
    inputs = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(128, activation='relu')(x)  # This can be the feature layer

    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    observer_network = build_waveform_observer()
    observer_network.summary()
