#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explore and visualize potential features.

Author:   
    Erik Johannes Husom

Created:  
    2020-10-15

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
import yaml

from config import DATA_FEATURIZED_PATH, DATA_PATH
from preprocess_utils import read_csv, move_column


def explore_features():
    """Clean up inputs and add features to data set.

    Args:
        filepaths (list of str): List of paths to files to process.

    """

    stage = "restructured"

    data_dir = "assets/data/" + stage + "/"

    filepaths = os.listdir(data_dir)

    # for filepath in filepaths:
    for i in range(1):

        filepath = data_dir + filepaths[8]

        # Read csv, and delete specified columns
        df, index = read_csv(
            filepath,
            delete_columns=["time", "calories"]
        )

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name="power", new_idx=0)

        #=====================================================================
        # Add features

        # ribcage_max = df["ribcage"].rolling(win).max()
        # df["gradient_ribcage"] = np.gradient(df["ribcage"])
        # df["diff_ribcage"] = df["ribcage"].diff()

        # Find peaks in breathing
        ribcage_peaks_indices = find_peaks(df["ribcage"], distance=10)[0]
        ribcage_peaks = np.zeros(len(df["ribcage"]))
        ribcage_peaks[ribcage_peaks_indices] = 1
        df["ribcage_peaks"] = ribcage_peaks

        abdomen_peaks_indices = find_peaks(df["abdomen"], distance=10)[0]
        abdomen_peaks = np.zeros(len(df["abdomen"]))
        abdomen_peaks[abdomen_peaks_indices] = 1
        df["abdomen_peaks"] = abdomen_peaks

        # Find breathing frequency
        win = 200 # deciseconds
        # Sum up the number of peaks (breaths) over the window, divide by
        # window size, which gives us breaths per second, and then multiply by
        # 600 deciseconds to get breaths per minute
        df["ribcage_freq"] = df["ribcage_peaks"].rolling(win).sum() * 600/win
        df["abdomen_freq"] = df["abdomen_peaks"].rolling(win).sum() * 600/win

        # FFT
        # df["ribcage_fft"] = np.fft.fft(df["ribcage"])
        ribcage_fft = np.fft.fft(df["ribcage"])
        plt.plot(ribcage_fft)

        plt.show()

        # Plot data frame with plotly as backend
        pd.options.plotting.backend = "plotly"
        fig = df.plot()

        fig.add_trace(go.Scatter(
            x=ribcage_peaks_indices,
            y=[df["ribcage"][j] for j in ribcage_peaks_indices],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='cross'
            ),
            name='Ribcage Peaks'
        ))

        fig.add_trace(go.Scatter(
            x=abdomen_peaks_indices,
            y=[df["abdomen"][j] for j in abdomen_peaks_indices],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='cross'
            ),
            name='Abdomen Peaks'
        ))

        fig.show()


if __name__ == '__main__':

    explore_features()
