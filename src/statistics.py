#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print statistics about data set.

Author:   
    Erik Johannes Husom

Created:  
    2020-10-13

"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_PATH
from preprocess_utils import read_csv, move_column

plt.style.use("ggplot")
WIDTH = 9
HEIGHT = 6

def statistics():
    """Print statistics about data set."""

    data_dir = "assets/data/restructured/"

    filepaths = os.listdir(data_dir)

    dfs = []

    for filepath in filepaths:

        filepath = data_dir + filepath

        df = pd.read_csv(filepath, index_col=0)

        dfs.append(df)

    # Merge all workouts to one dataframe
    merged_df = pd.concat(dfs, ignore_index=True)


    # Power histogram
    plt.subplot(221)
    merged_df.power.hist(bins=50)
    plt.xlim([0,1250])
    plt.xlabel("power (W)")
    # plt.title("Power histogram")

    # Heart rate histogram
    plt.subplot(222)
    merged_df.heartrate.hist(bins=50)
    plt.xlim([25,180])
    plt.xlabel("heart rate (bpm)")
    # plt.title("Heart rate histogram")

    # Ribcage histogram
    plt.subplot(223)
    merged_df.ribcage.hist(bins=50)
    plt.xlim([1000,2750])
    plt.xlabel("RIP ribcage (mV)")
    # plt.title("Ribcage histogram")
    
    # Abdomen histogram
    plt.subplot(224)
    merged_df.abdomen.hist(bins=50)
    plt.xlim([0,2250])
    plt.xlabel("RIP abdomen (mV)")
    # plt.title("Abdomen histogram")

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "histogram_training.pdf")
    plt.show()

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="Visualize data set")

    # parser.add_argument("-r", "--restructured", help="Plot restructured data.",
    #         action="store_true")
    # parser.add_argument("-f", "--featurized", help="Plot featurized data.",
    #         action="store_true")

    # args = parser.parse_args()

    # if args.restructured:
    #     visualize("restructured")

    # if args.featurized:
    #     visualize("featurized")

    statistics()


