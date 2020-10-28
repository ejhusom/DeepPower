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
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from tensorflow.keras import models
import yaml

from config import METRICS_FILE_PATH, PLOTS_PATH, PREDICTION_PLOT_PATH, DATA_PATH


def evaluate(model_filepath, test_filepath):
    """Evaluate model to estimate power.

    Args:
        model_filepath (str): Path to model.
        test_filepath (str): Path to test set.

    """

    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    test = np.load(test_filepath)

    X_test = test["X"]
    y_test = test["y"]

    model = models.load_model(model_filepath)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print("MSE: {}".format(mse))

    plot_prediction(y_test, y_pred, inputs=X_test, info="(MSE: {})".format(mse))

    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(dict(mse=mse), f)


def plot_prediction(y_true, y_pred, inputs=None, info="", backend="plotly"):
    """Plot the prediction compared to the true targets.

    A matplotlib version of the plot is saved, while a plotly version by
    default is shown. To show the plot with matplotlib instead, the 'backend'
    parameter has to be changed to 'matplotlib'.

    Args:
        y_true (array): True targets.
        y_pred (array): Predicted targets.
        include_input (bool): Whether to include inputs in plot. Default=True.
        inputs (array): Inputs corresponding to the targets passed. If
            provided, the inputs will be plotted together with the targets.
        info (str): Information to include in the title string.
        backend (str): Whether to use matplotlib or plotly as plot backend.
            Default='plotly'.

    """

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("time step")
    ax1.set_ylabel("power (W)")

    ax1.plot(y_true, label="true")
    ax1.plot(y_pred, label="pred")

    if inputs is not None:
        input_columns = pd.read_csv(DATA_PATH / "input_columns.csv")
        num_features = inputs.shape[-1]

        ax2 = ax1.twinx()
        ax2.set_ylabel("scaled units")

        for i in range(num_features):
            ax2.plot(inputs[:, -1, i], label=input_columns.iloc[i+1,1])
         

    fig.legend()

    plt.title("True vs pred " + info, wrap=True)
    plt.savefig(PREDICTION_PLOT_PATH)

    if backend == "matplotlib":
        plt.show()
    else:

        x = np.linspace(0, len(y_true)-1, len(y_true))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        config = dict({'scrollZoom': True})

        fig.add_trace(
                go.Scatter(x=x, y=y_true.reshape(-1), name="true"),
                secondary_y=False,
        )

        fig.add_trace(
                go.Scatter(x=x, y=y_pred.reshape(-1), name="pred"),
                secondary_y=False,
        )

        if inputs is not None:
            input_columns = pd.read_csv(DATA_PATH / "input_columns.csv")
            num_features = inputs.shape[-1]

            for i in range(num_features):

                fig.add_trace(
                        go.Scatter(
                            x=x, y=inputs[:, -1, i],
                            name=input_columns.iloc[i+1, 1]
                        ),
                        secondary_y=True,
                )

        fig.update_layout(title_text="True vs pred " + info)
        fig.update_xaxes(title_text="time step")
        fig.update_yaxes(title_text="power (W)", secondary_y=False)
        fig.update_yaxes(title_text="scaled units", secondary_y=True)

        fig.write_html(str(PLOTS_PATH / "prediction.html"))
        fig.show(config=config)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        try:
            evaluate("assets/models/model.h5", "assets/data/scaled/test.npz")
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        evaluate(sys.argv[1], sys.argv[2])
