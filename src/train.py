#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-16  

"""
import sys
import time

import numpy as np
import yaml

from config import MODELS_PATH, MODELS_FILE_PATH
from model import cnn


def train(filepath):
    """Train model to estimate power.

    Args:
        filepath (str): Path to training set.

    """

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

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

    time_id = time.strftime("%Y%m%d%H%M%S")

    # model.save(MODELS_PATH / (time_id + ".h5"))
    model.save(MODELS_FILE_PATH)


if __name__ == "__main__":

    train(sys.argv[1])
