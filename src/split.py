#!/usr/bin/env python3
# ============================================================================
# File:     split.py
# Author:   Erik Johannes Husom
# Created:  2020-09-16
# ----------------------------------------------------------------------------
# Description:
# Split data set into training and test set.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml

from config import Config
from preprocess_utils import *
from utils import *



def combine(filepaths):
    """Combine data from multiple workouts into one dataset.

    Args:
        filepaths (list of str): A list of paths to files containing
            sequentialized data.

    Returns:
        X (array): Input array.
        y ( array): Output/target array.

    """
        

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    inputs = []
    outputs = []

    for filepath in filepaths:
        infile = np.load(filepath)
        
        inputs.append(infile["X"])
        outputs.append(infile["y"])
        
    X = np.concatenate(inputs)
    y = np.concatenate(outputs)

    return X, y


def split(X, y):

    Config.DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["split"]

    train_split = params["train_split"]

    train_elements = int(X.shape[0]*train_split)

    # Split X and y into train and test
    X_train, X_test = np.split(X, [train_elements])
    y_train, y_test = np.split(y, [train_elements])

    # Save train and test data into a binary file
    np.savez(
        Config.DATA_SPLIT_PATH / "train.npz", 
        X_train=X_train, y_train=y_train
    )
    np.savez(
        Config.DATA_SPLIT_PATH / "test.npz", 
        X_test=X_test, y_test=y_test
    )


if __name__ == "__main__":

    X, y = combine(sys.argv[1:])
    split(X, y)



