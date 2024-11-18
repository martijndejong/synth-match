"""
DEBUG / DEVELOPMENT CODE
This observer directly passes the synthesizer parameters error to the agent network.
The purpose of this observer is to debug the agent logic in a simplified setup (instead of end-to-end training).
"""

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def build_parameter_observer(input_shape=(2,)):
    inputs = Input(shape=input_shape)

    # # Directly use the inputs as outputs
    # model = Model(inputs=inputs, outputs=inputs)

    # Bring output to higher dimensionality to simulate feature dimensionality of actual observer network
    outputs = Dense(128, activation='relu')(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    observer_network = build_parameter_observer()
    observer_network.summary()
