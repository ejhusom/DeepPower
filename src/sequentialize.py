#!/usr/bin/env python3
# ============================================================================
# File:     seuqentialize.py
# Author:   Erik Johannes Husom
# Created:  2020-09-16
# ----------------------------------------------------------------------------
# Description:
# Split data into input/output sequences.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml

from config import Config
from preprocess_utils import *
from utils import *


def sequentialize(filepaths):

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    Config.DATA_SEQUENTIALIZED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]

    hist_size = params["hist_size"]

    for filepath in filepaths:

        df, index = read_csv(filepath)

        # Convert to numpy
        data = df.to_numpy()

        # Split into input (X) and output/target (y)
        X = data[:, 1:].copy()
        y = data[:, 0].copy().reshape(-1, 1)

        # Combine y and X to get correct format for sequentializing
        data = np.hstack((y, X))

        # Split into sequences
        X, y = split_sequences(data, hist_size)

        # Save X and y into a binary file
        np.savez(
            Config.DATA_SEQUENTIALIZED_PATH
            / (
                os.path.basename(filepath).replace(
                    "featurized.csv", "sequentialized.npz"
                )
            ),
            X=X,
            y=y,
        )

if __name__ == "__main__":

    sequentialize(sys.argv[1:])
