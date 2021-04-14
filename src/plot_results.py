#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")
WIDTH = 9
HEIGHT = 6


def plot_results_lstm():

    df = pd.read_csv("assets/results/lstm_hidden_units_model3.csv")

    plt.figure(figsize=(WIDTH,HEIGHT))
    plt.subplot(211)
    plt.plot(df["num_hidden_units"], df["mse"], ".-")
    plt.ylabel("MSE")
    plt.xticks([10,20,30,40,50,60,70])
    plt.subplot(212)
    plt.plot(df["num_hidden_units"], df["r2"], ".-", color=(74/256,137/256,185/256))
    plt.xticks([10,20,30,40,50,60,70])
    plt.ylabel("R2 score")
    plt.xlabel("Num. of hidden units")
    plt.savefig("assets/plots/lstm_hidden_units.pdf")
    plt.show()



if __name__ == '__main__':

    plot_results_lstm()
