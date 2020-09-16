#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scaling data.

Scaling the inputs of data set.

TODO:
    Implement scaling.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-16

"""

from config import DATA_SPLIT_PATH


def scale(train_file, test_file):
    """Scale train and test set.

    Args:
        train_file (str): Path to file containing train set.
        test_file (str): Path to file containing test set.

    """


    train = np.load(train_file)
    test = np.load(test_file)

    X_train = train["X"]
    X_test = test["X"]
    y_train = train["y"]
    y_test = test["y"]



