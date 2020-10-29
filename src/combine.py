#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combines workout files into one.

Author:   
    Erik Johannes Husom

Created:  
    2020-10-29

"""
import numpy as np

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

if __name__ == "__main__":

    X, y = combine(sys.argv[1:])
