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

from config import METRICS_PATH


def evaluate(model_filepath, test_filepath):
    """Evaluate model to estimate power.

    Args:
        model_filepath (str): Path to model.
        test_filepath (str): Path to test set.

    """

    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    # Load training set
    test = np.load(test_filepath)

    X_test = test["X"]
    y_test = test["y"]

    model = models.load_model(model_filepath)

    # r_squared = model.scores(X_test, y_test)

    y_pred = model.predict(X_test)
    plot_prediction(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)

    time_id, _ = os.path.splitext(os.path.basename(model_filepath))

    with open(METRICS_PATH / (time_id + "-metrics.json"), "w") as f:
        json.dump(dict(rmse=rmse), f)
        # json.dump(dict(r_squared=r_squared, rmse=rmse), f)


def plot_prediction(y_true, y_pred, include_input=True):
    """Plot the prediction compared to the true targets.

    Args:
        y_true (array): True targets.
        y_pred (array): Predicted targets.
        include_input (bool): Whether to include inputs in plot. Default=True.

    """

    plt.figure()

    plt.plot(y_true, label="true")
    plt.plot(y_pred, label="pred")

    # if include_input:
    #     # X = np.load("./assets/data/featurized/")
    #     for i in range(X_test_pre_seq.shape[1]):
    #         # plt.plot(df.iloc[:,i], label=input_columns[i])
    #         plt.plot(
    #             X_test_pre_seq[:, i] * 250, label=input_columns[i + 1]
    #         )

    plt.legend()
    plt.title("True vs pred", wrap=True)
    plt.autoscale()
    # plt.savefig(result_dir + time_id + "-pred.png")
    plt.show()


if __name__ == "__main__":

    evaluate(sys.argv[1], sys.argv[2])
