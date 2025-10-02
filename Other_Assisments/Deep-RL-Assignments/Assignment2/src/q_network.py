import os
import copy
import json
import keras
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.layers import Dense
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
class QNetwork:
    def __init__(self, args, input, output, learning_rate):
        self.weights_path = "models/%s/%s" % (
            args.env,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if args.model_file is None:

            self.model = keras.models.Sequential()
            self.model.add(
                Dense(
                    128,
                    activation="relu",
                    input_dim=input,
                    kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                )
            )
            self.model.add(
                Dense(
                    128,
                    activation="relu",
                    kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                )
            )
            self.model.add(
                Dense(
                    128,
                    activation="relu",
                    kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                )
            )
            self.model.add(
                Dense(
                    output,
                    activation="linear",
                    kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                )
            )
            adam = keras.optimizers.Adam(lr=learning_rate)
            self.model.compile(
                loss="mean_squared_error", optimizer=adam, metrics=["accuracy"]
            )
        else:
            print("Loading pretrained model from", args.model_file)
            self.load_model_weights(args.model_file)

    def save_model_weights(self, step):

        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        self.model.save(os.path.join(self.weights_path, "model_%d.h5" % step))

    def load_model_weights(self, weight_file):

        self.model = keras.models.load_model(weight_file)