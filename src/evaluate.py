#!/usr/bin/env python3
"""Evaluate deep learning model to estimate power from breathing data.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-17

"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import models
import yaml

from config import METRICS_FILE_PATH, PREDICTION_PLOT_PATH


def evaluate(model_filepath, test_filepath):
    """Evaluate model to estimate power.

    Args:
        model_filepath (str): Path to model.
        test_filepath (str): Path to test set.

    """

    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load training set
    test = np.load(test_filepath)

    X_test = test["X"]
    y_test = test["y"]

    model = models.load_model(model_filepath)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)

    plot_prediction(y_test, y_pred)

    # time_id, _ = os.path.splitext(os.path.basename(model_filepath))
    # with open(METRICS_PATH / (time_id + "-metrics.json"), "w") as f:
    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(dict(rmse=rmse), f)


def plot_prediction(y_true, y_pred, include_input=True):
    """Plot the prediction compared to the true targets.

    Args:
        y_true (array): True targets.
        y_pred (array): Predicted targets.
        include_input (bool): Whether to include inputs in plot. Default=True.

    """

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    plt.plot(y_true, label="true")
    plt.plot(y_pred, label="pred")

    plt.legend()
    plt.title("True vs pred", wrap=True)
    plt.autoscale()
    # plt.savefig(result_dir + time_id + "-pred.png")
    plt.show()

    with open(PREDICTION_PLOT_PATH, "w") as f:
        json.dump({"prediction": [{
                "y_true": t,
                "y_pred": p
            } for t, p in zip(y_true, y_pred)
        ]}, f)


if __name__ == "__main__":

    evaluate(sys.argv[1], sys.argv[2])
