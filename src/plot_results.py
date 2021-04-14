#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")
WIDTH = 9
HEIGHT = 6


def plot_results_lstm(model_num):

    df = pd.read_csv(f"assets/results/lstm_hidden_units_model{model_num}.csv")

    plt.figure(figsize=(WIDTH,HEIGHT))
    plt.subplot(211)
    plt.plot(df["num_hidden_units"], df["mse"], ".-")
    plt.ylabel("MSE")
    plt.xticks([10,20,30,40,50,60,70,80,90,100])
    plt.subplot(212)
    plt.plot(df["num_hidden_units"], df["r2"], ".-", color=(74/256,137/256,185/256))
    plt.xticks([10,20,30,40,50,60,70,80,90,100])
    plt.ylabel("R2 score")
    plt.xlabel("Num. of hidden units")
    plt.savefig(f"assets/plots/lstm_hidden_units_model{model_num}.pdf")
    plt.show()


def plot_results_over_hist_size(model, model_num):

    df = pd.read_csv(f"assets/results/{model}_history_size_model{model_num}.csv")

    plt.figure(figsize=(WIDTH,HEIGHT))
    plt.subplot(211)
    plt.plot(df["hist_size"], df["mse"], ".-")
    plt.ylabel("MSE")
    plt.xticks([20,40,60,80,100,120])
    plt.subplot(212)
    plt.plot(df["hist_size"], df["r2"], ".-", color=(74/256,137/256,185/256))
    plt.xticks([20,40,60,80,100,120])
    plt.ylabel("R2 score")
    plt.xlabel("Window size (no. of samples)")
    plt.savefig(f"assets/plots/{model}_hidden_units_model{model_num}.pdf")
    plt.show()

if __name__ == '__main__':

    # plot_results_lstm(3)
    # plot_results_lstm(9)
    plot_results_over_hist_size("cnn", 6)
    plot_results_over_hist_size("cnn", 10)
    plot_results_over_hist_size("dnn", 3)

