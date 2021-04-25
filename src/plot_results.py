#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")
WIDTH = 9
HEIGHT = 4


def plot_results_lstm(model_num):

    df = pd.read_csv(f"assets/results/lstm_hidden_units_model{model_num}.csv")

    plt.figure(figsize=(WIDTH,HEIGHT))
    # plt.subplot(211)
    plt.plot(df["num_hidden_units"], df["mse"], ".-")
    plt.ylabel("MSE")
    plt.xticks([10,20,30,40,50,60,70,80,90,100,110])
    # plt.subplot(212)
    # plt.plot(df["num_hidden_units"], df["r2"], ".-", color=(74/256,137/256,185/256))
    # plt.xticks([10,20,30,40,50,60,70,80,90,100,110])
    # plt.ylabel("R2 score")
    plt.xlabel("Num. of hidden units")
    plt.savefig(f"assets/plots/lstm_hidden_units_model{model_num}.pdf")
    plt.show()

def plot_results_cnn(model_num):

    df = pd.read_csv(f"assets/results/cnn_kernel_size_model{model_num}.csv")

    xticks = [2,4,6,8,10]

    plt.figure(figsize=(WIDTH,HEIGHT))
    plt.subplot(211)
    plt.plot(df["kernel_size"], df["mse"], ".-")
    plt.ylabel("MSE")
    plt.xticks(xticks)
    plt.subplot(212)
    plt.plot(df["kernel_size"], df["r2"], ".-", color=(74/256,137/256,185/256))
    plt.xticks(xticks)
    plt.ylabel("R2 score")
    plt.xlabel("Kernel size")
    plt.savefig(f"assets/plots/cnn_kernel_size_model{model_num}.pdf")
    plt.show()

def plot_results_over_hist_size():

    models = ["cnn", "dnn", "lstm"]
    model_num = [6, 3, 3]

    xticks = [20,40,60,80,100,120,140]

    plt.figure(figsize=(WIDTH,HEIGHT))

    for m, i in zip(models, model_num):

        df = pd.read_csv(f"assets/results/{m}_history_size_model{i}.csv")

        plt.plot(df["hist_size"], df["mse"], ".-", 
                label=f"{m.upper()}, feature set {i}"
        )
        plt.ylabel("MSE")
        plt.xticks(xticks)
        plt.xlabel("History size (no. of time steps)")

    plt.legend()
    plt.savefig(f"assets/plots/history_size.pdf")
    plt.show()

if __name__ == '__main__':

    plot_results_lstm(3)
    # plot_results_over_hist_size()
    # plot_results_cnn(6)

