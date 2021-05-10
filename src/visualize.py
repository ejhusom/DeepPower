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
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import MODELS_PATH
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

        fig = df.plot(title=filepath)

        if backend == "plotly":
            fig.show()
        else:
            plt.title(filepath)
            plt.show()

def plot_example_workouts():

    stage = "examples"

    data_dir = "assets/data/" + stage + "/"

    paths = sorted(os.listdir(data_dir))

    filepaths = []

    for p in paths:
        if p.endswith(".csv"):
            filepaths.append(p)


    steadystate = np.ones(12000)*200
    intervals = np.concatenate([
            np.ones(600)*100,
            np.ones(3000)*300,
            np.ones(1300)*100,
            np.ones(1800)*300,
            np.ones(1200)*100,
            np.ones(2400)*300,
            np.ones(1700)*100
            ])
    ramp = np.concatenate([
            np.ones(3000)*100,
            np.ones(3000)*150,
            np.ones(2400)*200,
            np.ones(1800)*250,
            np.ones(1200)*300,
            np.ones(600)*350,
    ])

    workoutplans = [
            steadystate,
            intervals,
            ramp
    ]

    workout_labels = ["(a)", "(b)", "(c)"]

    fig = plt.figure(figsize=(WIDTH,7))
    ax = None

    for i, filepath in enumerate(filepaths):

        filepath = data_dir + filepath

        # Read csv, and delete specified columns
        df = pd.read_csv(filepath, index_col=0)

        t0 = 0

        t = (df["time"] - df["time"].iloc[0]) / 60

        if i == 0:
            ax = fig.add_subplot(3,1,i+1)
        else:
            ax = fig.add_subplot(3,1,i+1, sharex = ax, sharey = ax)


        ax.set_ylim([0, 400])
        ax.set_xlabel("time (min)")
        l1 = ax.plot(t[t0:t0+12000], workoutplans[i][t0:t0+12000], alpha=0.8,
                label="planned structure")
        l2 = ax.plot(t[t0:t0+12000], df["power"].iloc[t0:t0+12000], 
                label="actual structure") 

        ax.set_title(workout_labels[i], loc="left")

        # if i == 1:
        ax.set_ylabel("power (W)")

    fig.legend( labels=["planned", "actual"], loc="center right")

    plt.subplots_adjust(right=0.85, hspace=0.9)
    plt.savefig("assets/plots/workout_examples.pdf")
    plt.show()


def plot_prediction():

    y = np.load("assets/plots/evaluated.npz")
    scaler = pickle.load(open("outputscaler.pkl", "rb"))
    
    # Read data
    y_test = y["y_test"]
    y_pred = y["y_pred"]

    # Rescale data
    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(mape)

    # Pick suitable amount of data
    n = 12000
    y_test = y_test[-n:]
    y_pred = y_pred[-n:]

    l = len(y_test)

    t = np.linspace(0,(l-1)/10/60,l)

    plt.figure(figsize=(WIDTH,HEIGHT))
    plt.plot(t, y_test, label="true")
    plt.plot(t, y_pred, label="predicted")
    plt.xlabel("time (min)")
    plt.ylabel("power (W)")
    plt.legend()
    plt.savefig("assets/plots/final_result.pdf")
    plt.show()


def mean_absolute_percentage_error(y_true, y_pred):

    m = np.sum(np.abs((y_true - y_pred) / y_true)) / len(y_true)

    return m


def plot_example_data():

    stage = "examples"

    data_dir = "assets/data/" + stage + "/"

    paths = sorted(os.listdir(data_dir))

    filepaths = []

    for p in paths:
        if p.endswith(".csv"):
            filepaths.append(p)


    workout_labels = ["(a)", "(b)", "(c)", "(d)"]
    data_labels = ["RIP ribcage (mV)", 
            "RIP abdomen (mV)",
            "Heart rate (bpm)",
            "Power (W)"]

    cols = ["ribcage", "abdomen", "heartrate", "power"]

    fig = plt.figure(figsize=(WIDTH,10))
    ax = None

    filepath = data_dir + filepaths[1]
    df = pd.read_csv(filepath, index_col=0)

    t0 = 0

    t = (df["time"] - df["time"].iloc[0]) / 60

    for i, col in enumerate(cols):

        if i == 0:
            ax = fig.add_subplot(4,1,i+1)
        else:
            ax = fig.add_subplot(4,1,i+1, sharex = ax)
            # ax = fig.add_subplot(4,1,i+1, sharex = ax, sharey = ax)


        # ax.set_ylim([0, 400])
        ax.set_xlabel("time (min)")
        l1 = ax.plot(t[t0:t0+12000], df[col].iloc[t0:t0+12000], 
                label=data_labels[i]) 

        ax.set_title(workout_labels[i], loc="left")

        ax.set_ylabel(data_labels[i])

    # fig.legend(labels=["planned", "actual"], loc="center right")

    plt.subplots_adjust(right=0.85, hspace=0.9)
    plt.savefig("assets/plots/data_example.pdf")
    plt.show()

def plot_cdrift():

    stage = "cardiovascular-drift"

    data_dir = "assets/data/" + stage + "/"

    paths = sorted(os.listdir(data_dir))

    filepaths = []

    for p in paths:
        if p.endswith(".csv"):
            filepaths.append(p)


    data_labels = [
            "Heart rate (bpm)",
            "Speed (kph)"]

    cols = ["hr", "kph"]

    fig = plt.figure(figsize=(8,4))
    ax = None

    filepath = data_dir + filepaths[0]
    df = pd.read_csv(filepath)

    t0 = 1185
    t1 = 550

    t = df["secs"] / 60
    df["kph"] = df["kph"] - df["kph"] + 7.9

    for i, col in enumerate(cols):

        if i == 0:
            ax = fig.add_subplot(2,1,i+1)
        else:
            ax = fig.add_subplot(2,1,i+1, sharex = ax)

        ax.set_xlabel("time (min)")
        l1 = ax.plot(
                t[t0:-t1], 
                df[col].iloc[t0:-t1], 
                label=data_labels[i]
        ) 
        # l1 = ax.plot(t[t0:t0+12000], df[col].iloc[t0:t0+12000],
        #         label=data_labels[i]) 

        ax.set_ylabel(data_labels[i])

    # fig.legend(labels=["planned", "actual"], loc="center right")

    plt.subplots_adjust(right=0.85, hspace=0.9)
    plt.savefig("assets/plots/cardiovascular_drift.pdf")
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
    parser.add_argument("-p", "--prediction", help="Plot final prediction",
            action="store_true")
    parser.add_argument("-d", "--exampledata", help="Plot example data",
            action="store_true")
    parser.add_argument("-c", "--cdrift", help="Plot cardiovascular drift",
            action="store_true")

    args = parser.parse_args()

    if args.restructured:
        visualize("restructured", backend=args.backend)

    if args.featurized:
        visualize("featurized", backend=args.backend)
    
    if args.examples:
        plot_example_workouts()

    if args.exampledata:
        plot_example_data()

    if args.prediction:
        plot_prediction()

    if args.cdrift:
        plot_cdrift()
