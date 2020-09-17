#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-16  

"""
import yaml

from model import cnn


def build_model():
    pass


def train(filepath):
    """Train model to estimate power.

    Args:
        filepath (str): Path to training set.

    """

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["train"]

    # Load training set
    train = np.load(filepath)

    X_train = train["X"]
    y_train = train["y"]

    hist_size = X_train.shape[-2]
    n_features = X_train.shape[-1]

    # Build model
    model = cnn(hist_size, n_features)

    print(model.summary())

    history = model.fit(
        X_train, y_train, epochs=params["n_epochs"], batch_size=params["batch_size"]
    )
