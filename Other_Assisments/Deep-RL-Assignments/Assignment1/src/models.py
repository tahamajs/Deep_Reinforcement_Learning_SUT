# Author: Taha Majlesi - 810101504, University of Tehran

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def make_model():
    model = Sequential()
    model.add(Dense(10, activation="tanh", input_dim=4))
    model.add(Dense(2, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model
