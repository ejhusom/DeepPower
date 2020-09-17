#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creating deep learning model for estimating power from breathing.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model


def cnn(input_x, input_y, n_steps_out=1):
    """Define a CNN model architecture using Keras.

    Args:
        input_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_y (int): Number of features for each time step in the input data.
        n_steps_out (int): Number of output steps.

    Returns:
        model (keras model): Model to be trained.

    """

    kernel_size = 2

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(input_x, input_y),
        )
    )
    model.add(layers.Conv1D(filters=32, kernel_size=kernel_size, activation="relu"))
    # model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(n_steps_out, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mape"])

    return model
