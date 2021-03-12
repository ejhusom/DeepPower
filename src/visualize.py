#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize data.

Author:   
    Erik Johannes Husom

Created:  
    2020-09-30

"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocess_utils import read_csv, move_column

plt.style.use("ggplot")
WIDTH = 9
HEIGHT = 6

def visualize(stage="restructured", backend="plotly"):
    """Visualize data set.

    Args:
        stage (str): Which stage of the data to plot. Options:
            - restructured
            - featurized

    """

    data_dir = "assets/data/" + stage + "/"

    filepaths = os.listdir(data_dir)

    if backend == "plotly":
        pd.options.plotting.backend = "plotly"

    for filepath in filepaths:

        filepath = data_dir + filepath

        # Read csv, and delete specified columns
        df = pd.read_csv(filepath, index_col=0)

        fig = df.plot()

        if backend == "plotly":
            fig.show()
        else:
            # plt.title(filepath)
            plt.show()

def plot_example_workouts():

    stage = "examples"

    data_dir = "assets/data/" + stage + "/"

    filepaths = os.listdir(data_dir)

    fig = plt.figure(figsize=(WIDTH,HEIGHT))
    # axes = plt.gca()
    # axes.set_ylim([0, 400])
    ax = None

    for i, filepath in enumerate(filepaths):

        if not filepath.endswith(".csv"):
            continue

        filepath = data_dir + filepath

        # Read csv, and delete specified columns
        df = pd.read_csv(filepath, index_col=0)

        t0 = 0

        t = (df["time"] - df["time"].iloc[0]) / 60

        if i == 0:
            ax = fig.add_subplot(3,1,i)
        else:
            ax = fig.add_subplot(3,1,i, sharex = ax, sharey = ax)

        ax.set_ylabel("power (W)")

        ax.set_ylim([0, 400])
        ax.set_xlabel("time (min)")
        ax.plot(t[t0:t0+12000], df["power"].iloc[t0:t0+12000])
        # ax.title(filepath)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize data set")

    parser.add_argument("-r", "--restructured", help="Plot restructured data.",
            action="store_true")
    parser.add_argument("-f", "--featurized", help="Plot featurized data.",
            action="store_true")
    parser.add_argument("-b", "--backend", default="plotly",
        help="Backend, plotly or matplotlib")
    parser.add_argument("-e", "--examples", help="Plot example workouts",
            action="store_true")

    args = parser.parse_args()

    if args.restructured:
        visualize("restructured", backend=args.backend)

    if args.featurized:
        visualize("featurized", backend=args.backend)
    
    if args.examples:
        plot_example_workouts()
